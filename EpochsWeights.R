for(epoch in colnames(weights)) {
  png(paste('Python/networks/sentiment_analisys/weights/', epoch, ".png"))
  plot(weights[,epoch], ylab=paste("Weights", epoch))
  dev.off()
}

