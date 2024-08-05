tar -czf - best_cnn_model.pth | split -b 50M - best_cnn_model.pth.part_
