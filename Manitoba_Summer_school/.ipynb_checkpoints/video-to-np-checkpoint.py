import os
import numpy as np
import skvideo.io  
import imageio



def vid2np(vid):
    videodata = skvideo.io.vread(vid+'.avi') #as_grey=True[:,:,:,2]
    videodata = videodata[:,:,:,2]//2+videodata[:,:,:,0]//2
    sp = videodata.shape
    print(f'frames {sp[0]}, width {sp[1]}, length {sp[2]}')
    for i,elem in enumerate(videodata):
        imageio.imwrite(os.path.join('data', vid, str(i)  + '.tif'), elem)
    
if __name__ == "__main__":
    os.chdir('..')
    dirs = ["human_blood", "human_breast"]
    for dir in dirs:
        vid2np(dir)