
#create loc by year env column
for (x in length(locations)){
  loc = locations[[x]]
  newDF = pheno[(pheno$Location_code = loc),]
  assign(paste0(loc), newDF)
}

env = list()
for (x in 1:nrow(pheno)){
  
  newcol = paste(pheno[x,1],pheno[x,2],sep = "")
  env[[x]] = newcol
}
env = as.data.frame(env)
env = t(env)
data = cbind(env,pheno)

#create a unique df for each env
envList = list()
for (x in 1:nrow(envsDF)){
  x = envsDF[x,1]
  df = data[(pheno$env = x),]
  assign(paste0((x),df))
  envList[[x]] = df
}

x <- split(data,data$env)

environmentList=list()
total = length(x)
for (i in 1:total){
  environment = x[[i]]
  rownames = environment[,6]
  
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
  
  environmentNew = environment[,-c(2,3,4,5,7,9,12:17)]
  
  rownames(environmentNew) = rownames 
  environmentNew = as.data.frame(environmentNew)
  environmentList[[i]] = environmentNew
}
