pheno = read.csv("phenotypesRaw.csv")

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

# isolate desired factors and use CDBN IDs for rownames
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
  
  
  environmentNew = environment[,-c(2,3,4,5,7,9,12:17)]
  
  rownames(environmentNew) = rownames 
  environmentNew = as.data.frame(environmentNew)
  environmentList[[i]] = environmentNew
}

### environmentList contains one dataframe for every environment. 
### each data frame contains the env name, line ID, SY, SW, DM
