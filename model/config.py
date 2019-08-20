

def get_model(model_name):
    if model_name == 'v2':
        # Base version, same as in https://arxiv.org/abs/1907.10478
        return {
            'left': [(2,3,32,'r','relu'), (2,3,64,'r','relu'), (2,3,96,'r','relu'), (2,3,128,'r','relu'),
                    (1,3,96,'u','leaky'), (1,3,64,'u','leaky'), (1,3,32,'u','leaky'), (1,3,3,'u','none')],
            'right': [(1,5,32,'r','relu'), (1,5,32,'r','relu'), (1,5,32,'r','relu'), (1,5,3,'r','none')]
        }
    elif model_name == 'v3':
        # Modified version to support longer context
        return {
            'left': [(2,7,32,'r','relu'), (2,5,32,'r','relu'), (2,5,32,'r','relu'), (2,3,32,'r','relu'),
                     (2,3,32,'r','relu'), (2,3,64,'r','relu'), (2,3,96,'r','relu'), (2,3,128,'r','relu'),
                     (1,3,96,'u','leaky'), (1,3,64,'u','leaky'), (1,3,32,'u','leaky'), (1,3,32,'u','leaky'),
                     (1,3,32,'u','leaky'), (1,3,32,'u','leaky'), (1,3,32,'u','leaky'), (1,3,3,'u','none')],
            'right': [(1,5,32,'r','relu'), (1,5,32,'r','relu'), (1,5,32,'r','relu'), (1,5,32,'r','relu'), 
                      (1,5,32,'r','relu'), (1,5,32,'r','relu'), (1,5,32,'r','relu'),(1,5,3,'r','none')]
        }