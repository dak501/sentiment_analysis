
a<-read.csv('results-ds1.csv')
b<-read.csv('results-ds2.csv')
x<-read.csv('results-ds3.csv')
y<-read.csv('results-ds4.csv')
z<-read.csv('results-ds5.csv')


mean_1<- c(1,mean(a$bowRF),1,'bowRF')
mean_1b<-c(2,mean(a$bowNB),1,'bowNB')
mean_1c<-c(3,mean(a$tfRF) ,1,'tfRF')
mean_1d<-c(4,mean(a$tfNB) ,1,'tfNB')

mean_2<- c(5,mean(b$bowRF),2,'bowRF')
mean_2b<-c(6,mean(b$bowNB),2,'bowNB')
mean_2c<-c(7,mean(b$tfRF) ,2,'tfRF')
mean_2d<-c(8,mean(b$tfNB) ,2,'tfNB')

mean_3<- c(9,mean(x$bowRF),3,'bowRF')
mean_3b<-c(10,mean(x$bowNB),3,'bowNB')
mean_3c<-c(11,mean(x$tfRF) ,3,'tfRF')
mean_3d<-c(12,mean(x$tfNB) ,3,'tfNB')

mean_4<- c(13,mean(y$bowRF),4,'bowRF')
mean_4b<-c(14,mean(y$bowNB),4,'bowNB')
mean_4c<-c(15,mean(y$tfRF) ,4,'tfRF')
mean_4d<-c(16,mean(y$tfNB) ,4,'tfNB')

mean_5<- c(17,mean(z$bowRF),5,'bowRF')
mean_5b<-c(18,mean(z$bowNB),5,'bowNB')
mean_5c<-c(19,mean(z$tfRF) ,5,'tfRF')
mean_5d<-c(20,mean(z$tfNB) ,5,'tfNB')

df <-rbind(mean_1,mean_1b,mean_1c,mean_1d,
		mean_2,mean_2b,mean_2c,mean_2d,
		mean_3,mean_3b,mean_3c,mean_3d,
		mean_4,mean_4b,mean_4c,mean_4d,
		mean_5,mean_5b,mean_5c,mean_5d)
colnames(df) <- c('i','avg','DS','recipe')

df

library(ggplot2)

ggplot(data=df, aes(x=DS, y=as.numeric(avg))) +
  geom_line(aes(color=recipe,group=recipe))+   geom_point(aes(color=recipe))+scale_y_continuous()