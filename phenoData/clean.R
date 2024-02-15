pheno = read.csv("phenotypesRaw.csv")

env = list()
for (x in 1:nrow(pheno)){
  newcol = paste(pheno[x,1],pheno[x,2],sep = "")
  env[[x]] = newcol
}
env = as.data.frame(env)
env = t(env)
data = cbind(env,pheno)

data = data[,-c(2,3,4,5,7,9,12:17)] #select desired variables

x <- split(data,data$env)

# isolate desired factors and use CDBN IDs for rownames
environmentList=list()
total = length(x)
for (i in 1:total){
  environment = x[[i]]
  rownames = environment[,2]
  
  fix = which(rownames=="CDBN_090")
  if (length(fix) > 1) {
    change = fix[[1]]
    rownames[[change]] = "CDBN_090A"
  }
  
  fix2 = which(rownames=="CDBN_152")
  if (length(fix2) > 1) {
    change = fix2[[1]]
    rownames[[change]] = "CDBN_152A"
  }
  
  fix3 = which(rownames=="CDBN_217")
  if (length(fix3) > 1) {
    change = fix3[[1]]
    rownames[[change]] = "CDBN_217A"
  }
  
  fix4 = which(rownames=="CDBN_304")
  if (length(fix4) > 1) {
    change = fix4[[1]]
    rownames[[change]] = "CDBN_304A"
  }

  rownames(environment) = rownames 
  environment = as.data.frame(environment)
  if (sum(is.na(environment)) < 1){
    environmentList[[i]] = environment
  }
  
}

clean <- Filter(Negate(is.null), environmentList)

