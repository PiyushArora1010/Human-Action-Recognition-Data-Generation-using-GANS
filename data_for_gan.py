import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow

def view_skeleton(row):
    coor = row
    coor_s = coor.copy()
    coor = [coor[i-1] for i in range(1,46) if i%3 !=0]
    size = np.float32([coor_s[i-1] for i in range(1,46) if i%3 ==0])
    coor = [[coor[i],coor[i+1]] for i in range(0,len(coor),2) ]
    coor = np.array(coor)
    plt.scatter(coor[:,0], coor[:,1],s=size*0.04)
    plt.show()
    return coor,size

def data_for_gan(path = '..\Data\Florence\Florence_dataset_WorldCoordinates.txt', frames = 30):
    file = open(path,'r')
    data = [i for i in file]
    data = [list(map(float,i.split())) for i in data]
    df = pd.DataFrame()
    df[['activity no.','idk1','idk2']+['f'+str(i) for i in range(1,46)]] = data
    df['Action'] = df['idk2']
    df.drop('idk2', inplace=True, axis = 1)
    df['Human'] = df['idk1']
    df.drop('idk1', inplace=True, axis = 1)
    df_fps = pd.DataFrame( columns=df.columns)
    frames_per_action = []
    for i in range(1,216):
        frames_per_action.append(len(df[df['activity no.'] == i]))

    list_of_actions = []
    for ix,i in enumerate(frames_per_action):
        list_of_rows = [j for j in range(i)]
        list_of_rows = list_of_rows[:frames]
        while len(list_of_rows) < frames:
            random = np.random.randint(0, i)
            list_of_rows.append(random)
        list_of_actions.append(np.array(sorted(list_of_rows)))
    list_of_actions = np.array(list_of_actions)
    
    for action in range(list_of_actions.shape[0]):
        dfTemp = pd.DataFrame(columns=df.columns)
        dfAction = df[df['activity no.'] == action+1].sort_index().reset_index(drop = True)
        dfTemp = pd.concat([dfTemp ,dfAction.loc[list_of_actions[action],dfAction.columns]], ignore_index=True)
        df_fps = pd.concat([df_fps, dfTemp],ignore_index=True)

    df = df_fps

    xchannel = []
    for i in range(1,216):
        values = df[df['activity no.'] == i][['f'+str(j) for j in range(1,46) if j in [k for k in range(1, 46, 3)]]].values.flatten()
        xchannel.append(values)
    xchannel = np.array(xchannel)
    xchannel = xchannel.reshape(215, frames, 15)
    xchannel = np.transpose(xchannel,(0,2,1))
    ychannel = []
    for i in range(1,216):
        values = df[df['activity no.'] == i][['f'+str(j) for j in range(2, 46, 3)]].values.flatten()
        ychannel.append(values)
    ychannel = np.array(ychannel)
    ychannel = ychannel.reshape(215, frames, 15)
    ychannel = np.transpose(ychannel, (0,2,1))
    zchannel = []
    for i in range(1,216):
        values = df[df['activity no.'] == i][['f'+str(j) for j in range(3, 46, 3)]].values.flatten()
        zchannel.append(values)
    zchannel = np.array(zchannel)
    zchannel = zchannel.reshape(215, frames, 15)
    zchannel = np.transpose(zchannel, (0,2,1))
    finalData = np.zeros((215 , 15 , frames , 3))
    finalData[:,:,:,0] = xchannel
    finalData[:,:,:,1] = ychannel
    finalData[:,:,:,2] = zchannel
    # finalData = np.transpose(finalData , (0,3,1,2))
    y = []
    print(np.unique(df['Action']))
    for i in range(1,216):
        y.append(df[df['activity no.'] == i]["Action"].values[0] - 1)
    return finalData,np.array(y),df