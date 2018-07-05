import pickle

#Usefull function to save a model and be able to train it later

# Save a model and its optimizer in filename
def Save_model(model,optimizer,filename):
    Save_dic={}
    Save_dic['model']=model
    Save_dic['optimizer']=optimizer
    
    with open(filename, 'wb') as Save:
        pickle.dump(Save_dic, Save, protocol=pickle.HIGHEST_PROTOCOL)
        


# load a train model and its optimizer in a tuple (model,optimizer)
def Load_model(filename):
    with open(filename, 'rb') as Save:
        Save_dic = pickle.load(Save)
    
    return Save_dic['model'],Save_dic['optimizer']
    


# Save a model and its optimizer in filename
def Save_modelloss(model,optimizer,loss_history,filename):
    Save_dic={}
    Save_dic['model']=model
    Save_dic['optimizer']=optimizer
    Save_dic['loss_history']=loss_history
    
    with open(filename, 'wb') as Save:
        pickle.dump(Save_dic, Save, protocol=pickle.HIGHEST_PROTOCOL)
        


# load a train model and its optimizer in a tuple (model,optimizer)
def Load_modelloss(filename):
    with open(filename, 'rb') as Save:
        Save_dic = pickle.load(Save)
    
    return Save_dic['model'],Save_dic['optimizer'],Save_dic['loss_history']