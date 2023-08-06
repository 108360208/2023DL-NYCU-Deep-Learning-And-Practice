bug:儲存最佳weights時，每次要在load進來做predict，準確率都是訓練最後一次的準確率(未修改)
修改:透過deepcopy儲存，直接賦值每次epoch都會更新，因為兩個共享同一個記憶體位置
best_model = model.state_dict()=>
best_model = copy.deepcopy(model.state_dict())