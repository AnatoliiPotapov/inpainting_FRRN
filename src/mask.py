import numpy as np


def create_random_mask(width=500, height=600):
    mask = np.zeros((height, width))
    
    for _ in range(np.random.randint(1, 5)):
        mask_width = np.random.randint(100, 150)
        mask_height = np.random.randint(100, 150)
        
        mask_x = np.random.randint(0, width - mask_width)
        mask_y = np.random.randint(0, height - mask_height)
        
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        
    return mask