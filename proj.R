# create mean df

a<-read.csv('results-ds1.csv')
b<-read.csv('results-ds2.csv')
x<-read.csv('results-ds3.csv')
y<-read.csv('results-ds4.csv')
z<-read.csv('results-ds5.csv')

a

mean_1<- c(mean(a$bowRF),mean(a$bowNB),mean(a$tfRF),mean(a$tfNB))
mean_2<- c(mean(b$bowRF),mean(b$bowNB),mean(b$tfRF),mean(b$tfNB))
mean_3<- c(mean(x$bowRF),mean(x$bowNB),mean(x$tfRF),mean(x$tfNB))
mean_4<- c(mean(y$bowRF),mean(y$bowNB),mean(y$tfRF),mean(y$tfNB))
mean_5<- c(mean(z$bowRF),mean(z$bowNB),mean(z$tfRF),mean(z$tfNB))

df <-rbind(mean_1,mean_2,mean_3,mean_4,mean_5)
colnames(df) <- c('bowRF','bowNB','tfRF','tfNB')
df

df.shapi <- df
df.shapi_ <- df
for (i in 2:5) {
	df.shapi_ [1,i-1]<-(shapiro.test (a[[i]])$p.value > 0.05)
	df.shapi [1,i-1]<-shapiro.test (a[[i]])$p.value
}
for (i in 2:5) {
	df.shapi_ [2,i-1]<-(shapiro.test (b[[i]])$p.value > 0.05)
	df.shapi [2,i-1]<-shapiro.test (b[[i]])$p.value
}
for (i in 2:5) {
	df.shapi_ [3,i-1]<-(shapiro.test (x[[i]])$p.value > 0.05)
	df.shapi [3,i-1]<-shapiro.test (x[[i]])$p.value
}
for (i in 2:5) {
	df.shapi_ [4,i-1]<-(shapiro.test (y[[i]])$p.value > 0.05)
	df.shapi [4,i-1]<-shapiro.test (y[[i]])$p.value
}
for (i in 2:5) {
	df.shapi_ [5,i-1]<-(shapiro.test (z[[i]])$p.value > 0.05)
	df.shapi [5,i-1]<-shapiro.test (z[[i]])$p.value
}

df.welch 
# all shapiro tests indicate normal distributions for all 5 dataset's 4 columns

df.shapi
df.shapi_
c4<-c(1,2,3,4)
df.welch<-data.frame (bowRF=as.logical(c4),bowNB=as.logical(c4),tfRF=as.logical(c4),tfNB=as.logical(c4))
df.welch2<-data.frame (bowRF=as.logical(c4),bowNB=as.logical(c4),tfRF=as.logical(c4),tfNB=as.logical(c4))

df.welch3<-data.frame (bowRF=as.logical(c4),bowNB=as.logical(c4),tfRF=as.logical(c4),tfNB=as.logical(c4))

df.welch4<-data.frame (bowRF=as.logical(c4),bowNB=as.logical(c4),tfRF=as.logical(c4),tfNB=as.logical(c4))

df.welch5<-data.frame (bowRF=as.logical(c4),bowNB=as.logical(c4),tfRF=as.logical(c4),tfNB=as.logical(c4))


for (i in 2:5) for (j in 2:5) {
	df.welch [i-1,j-1]<-as.logical(t.test (a[[i]],a[[j]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
	df.welch2 [i-1,j-1]<-as.logical(t.test (b[[i]],b[[j]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
	df.welch3 [i-1,j-1]<-as.logical(t.test (x[[i]],x[[j]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
	df.welch4 [i-1,j-1]<-as.logical(t.test (y[[i]],y[[j]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
	df.welch5 [i-1,j-1]<-as.logical(t.test (z[[i]],z[[j]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
}
df.welch

df.welch2

df.welch3

df.welch4

df.welch5


c5<-c(FALSE,FALSE,FALSE,FALSE,FALSE)

df2.welch<-data.frame (d1=c5,d2=c5,d3=c5,d4=c5,d5=c5)
df2.welch2<-data.frame (d1=c5,d2=c5,d3=c5,d4=c5,d5=c5)
df2.welch3<-data.frame (d1=c5,d2=c5,d3=c5,d4=c5,d5=c5)
df2.welch4<-data.frame (d1=c5,d2=c5,d3=c5,d4=c5,d5=c5)


i=2
df2.welch [1,2]<-as.logical(t.test (a[[i]],b[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [1,3]<-as.logical(t.test (a[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [1,4]<-as.logical(t.test (a[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [1,5]<-as.logical(t.test (a[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [2,3]<-as.logical(t.test (b[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [2,4]<-as.logical(t.test (b[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [2,5]<-as.logical(t.test (b[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [3,4]<-as.logical(t.test (x[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [3,5]<-as.logical(t.test (x[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch [4,5]<-as.logical(t.test (y[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
print ("bowRF")
df2.welch

i=3
df2.welch2 [1,2]<-as.logical(t.test (a[[i]],b[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [1,3]<-as.logical(t.test (a[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [1,4]<-as.logical(t.test (a[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [1,5]<-as.logical(t.test (a[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [2,3]<-as.logical(t.test (b[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [2,4]<-as.logical(t.test (b[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [2,5]<-as.logical(t.test (b[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [3,4]<-as.logical(t.test (x[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [3,5]<-as.logical(t.test (x[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch2 [4,5]<-as.logical(t.test (y[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)

print ("bowNB")
df2.welch2

i=4
df2.welch3 [1,2]<-as.logical(t.test (a[[i]],b[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [1,3]<-as.logical(t.test (a[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [1,4]<-as.logical(t.test (a[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [1,5]<-as.logical(t.test (a[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [2,3]<-as.logical(t.test (b[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [2,4]<-as.logical(t.test (b[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [2,5]<-as.logical(t.test (b[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [3,4]<-as.logical(t.test (x[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [3,5]<-as.logical(t.test (x[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch3 [4,5]<-as.logical(t.test (y[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)

print ("tfRF")
df2.welch3

i=5
df2.welch4 [1,2]<-as.logical(t.test (a[[i]],b[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [1,3]<-as.logical(t.test (a[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [1,4]<-as.logical(t.test (a[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [1,5]<-as.logical(t.test (a[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [2,3]<-as.logical(t.test (b[[i]],x[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [2,4]<-as.logical(t.test (b[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [2,5]<-as.logical(t.test (b[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [3,4]<-as.logical(t.test (x[[i]],y[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [3,5]<-as.logical(t.test (x[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)
df2.welch4 [4,5]<-as.logical(t.test (y[[i]],z[[i]],alternative="two.sided",var.equal=FALSE)$p.value<0.05)

print ("tfNB")
df2.welch4


#}

