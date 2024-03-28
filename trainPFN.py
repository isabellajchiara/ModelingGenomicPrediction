

# Manual seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

min_training_samples = 100
max_training_samples = 500

prior = priorGenerator(geno,pheno)

num_batches = -(pheno.shape[0] // -10)

# Make a schedule of eval poses to batches
# We have to do this before the fact so that the data generator threads
# and training loop are on the same page about where the eval pos should be


print('Approximating y distribution...')
parfunc = lambda i: prior[i][1]
samples = np.concatenate(Parallel(n_jobs=cpu_count())(delayed(parfunc)(i) for i in range(10)))


class MSEFullSupportBarDistribution(FullSupportBarDistribution):
    def forward(self, logits, y):  # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        return (self.mean(logits) - y) ** 2

def get_bucket_limits(num_outputs:int, full_range:tuple=None, ys:torch.Tensor=None):
    assert (ys is not None) or (full_range is not None)
    if ys is not None:
        ys = ys.flatten()
        if len(ys) % num_outputs: ys = ys[:-(len(ys) % num_outputs)]
        print(f'Using {len(ys)} y evals to estimate {num_outputs} buckets. Cut off the last {len(ys) % num_outputs} ys.')
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min() and full_range[1] >= ys.max()
            full_range = torch.tensor(full_range)
        ys_sorted, ys_order = ys.sort(0)
        bucket_limits = (ys_sorted[ys_per_bucket-1::ys_per_bucket][:-1]+ys_sorted[ys_per_bucket::ys_per_bucket])/2
        print(full_range)
        bucket_limits = torch.cat([full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)],0)

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs
        bucket_limits = torch.cat([full_range[0] + torch.arange(num_outputs).float()*class_width, torch.tensor(full_range[1]).unsqueeze(0)], 0)

    assert len(bucket_limits) - 1 == num_outputs and full_range[0] == bucket_limits[0] and full_range[-1] == bucket_limits[-1]
    return bucket_limits

criteria = MSEFullSupportBarDistribution(borders=get_bucket_limits(50 ys=torch.tensor(samples)).to(device))bucket_means = (criteria.borders[:-1] + criteria.bucket_widths / 2).cpu().numpy()

model = AmortizedNeuralGP(geno.shape[1], n_out=1,
                          emb_size=500, num_layers=3,
                          hidden_dim=a2000, num_heads=1,
                          dropout=0.05)

# Stick all of this info into the model for reference
model.set_bucket_means(bucket_means)
model.set_loss_object(criteria)
model.set_num_tokens(500)
model.set_min_training_samples(min_training_samples)

if device.type == 'cuda':
    model = torch.nn.DataParallel(model)

model.to(device)
batchsize = 30
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
scheduler = get_cosine_schedule_with_warmup(optimizer, int(1000/ batchsize), num_batches)

running_loss = []
running_r = []

for i, (x, y) in enumerate(loader):
    # Truncate the data if it's not the right feature size. This allows us
    # to use cached feature sets at a lower resolution.
    if x.shape[2] > feature_max_size:
        warnings.warn('Truncating datapoints from {0} to {1} features'.format(x.shape[2], feature_max_size))
        x = x[:, :, :feature_max_size]

    eval_starts = eval_sched[i * batch_size:(i * batch_size) + batch_size]
    assert len(np.unique(eval_starts)) == 1
    eval_start = eval_starts[0]

    #  Truncate eval samples to fit into max pop size
    x, y = x[:, :pop_max_size], y[:, :pop_max_size]

    x = x.transpose(1, 0)
    y = y.transpose(1, 0)

    out = model(x.to(device), y.to(device), eval_start)
    loss = torch.mean(criteria(out[eval_start:], y[eval_start:].to(device)))
    loss = loss / 16 #default accumulation steps

    # pearson r (just for display)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    pred = [target.loss_object.mean(out[eval_start:, j, :]).detach().cpu().numpy().flatten() for j in range(batch_size)]
    gt = [y[eval_start:, j].numpy().flatten() for j in range(batch_size)]
    r = np.mean([pearsonr(p, g)[0] for p, g in zip(pred, gt)])

    print('loss: {0:.2f} r: {1:.2f} seen: {2} ({3:.2f}%) lr: {4:.7f}'.format(loss.detach().cpu().numpy(),
                                                                             r,
                                                                             i * batch_size,
                                                                             (i * batch_size) / num_samples * 100.,
                                                                             scheduler.get_last_lr()[0]))

    loss.backward()

    # Gradient accumulation
    if ((i + 1) % 16 #default accumulation steps == 0) or (i in [len(loader) - 1, 0]):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

   
now = datetime.now()
save_model(model, os.path.join(args.save_path, now.strftime('final-%d.%m.%Y.%H.%M.pt')))
