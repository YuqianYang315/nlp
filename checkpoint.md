
模型文件保存，前两中只可以infer和eval，最后一种可以继续训练

# save an load the model via state_dict
```
#save
path="state_dict_model.pt"
torch.save(model.state_dict,path)
#load
model =Net()
model.load_state_dict(torch.load(path))
model.eval
```

# save and load entire model
```
path="state_dict_model.pt"

#save
torch.save(net,path)
#load
model =Net()
model=torch.load(path)
model.eval()
```

# save the checkpoint在模型训练中,用于推理或者继续训练
把optmizer中的state_dic也保存下来(不仅保存模型的参数，优化器参数，还有loss，epoch等（相当于一个保存模型的文件夹）)
```
#save
checkpoint = {"model_state_dict": net.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
path_checkpoint = "./checkpoint_{}_epoch.pkl".format(epoch)
torch.save(checkpoint, path_checkpoint)

#load
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```
