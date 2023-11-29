geno = pd.read_csv("unstructuredGeno.csv")
resp = pd.read_csv("unstructuredPheno.csv")
data = resp.join(geno)

train = torch.tensor(data.values, dtype=torch.float)
x = train[:,0]
y = train[:,1:3548]


for X, y in data
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
