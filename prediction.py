from alphabet_mode_main import predict_labels_from_frames
import os
from os.path import join
from statistics import mode
from pandas import DataFrame
import pandas as pd
import time
from sklearn.metrics import classification_report

print("Choose a recognition model: \n1. Alphabets \n2. Words")

choice = input("Choose an option: ")

if choice == '1':
    video_path = './alphabet_videos/demo'
    video_list = os.listdir(video_path)
    
    frame_path = './frames/demo'
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    pred_array = []
    
    for video_name in video_list:
        
        if video_name == '.DS_Store':
            continue
        
        print("Running for " + video_name)
        file_path = join(video_path, video_name)
        
        os.system('python3 ./handtracking/detect_single_threaded.py --source=%s --video=%s --frame_path=%s' %(file_path, video_name, frame_path))
        
        test_data = join(frame_path, video_name)
        pred = predict_labels_from_frames(test_data)
        
        try:
            prediction = mode(pred)
        except:
            prediction = ''
        
        gold_label = video_name[0]
        print("\nTrue Value: " + video_name[0] + " Prediction: " + prediction)
        pred_array.append([prediction, gold_label])

    df = DataFrame (pred_array,columns=['pred','true'])
    print(classification_report(df.pred, df.true))
    df.to_csv(join(video_path, 'results.csv'))

if choice == '2':
    video_path = './word_videos/demo'
    video_list = os.listdir(video_path)
    
    frame_path = './frames/demo'
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    pred_array = []

    for video_name in video_list:
        
        if video_name == '.DS_Store':
            continue
        print("Running for " + video_name)
        file_path = join(video_path, video_name)
        
        os.system('python3 ./handtracking/detect_single_threaded.py --source=%s --video=%s --frame_path=%s' %(file_path, video_name, frame_path))
        
        pos_key = pd.read_csv('./pos_keypoints/' + video_name + '.csv')
        
        right_wrist = pos_key.rightWrist_x
        right_arm = pos_key.rightWrist_y
        left_wrist = pos_key.leftWrist_x
        left_arm = pos_key.leftWrist_y
        
        word = []
        
        for i in range(len(right_wrist)-1):
            if (right_wrist[i+1]-right_wrist[i] > 0.4) or (right_arm[i+1]-right_arm[i] > 0.4) or (left_wrist[i+1]-left_wrist[i] > 0.4) or (left_arm[i+1]-left_arm[i] > 0.4):
                till = i
            
            test_data = join(frame_path, video_name)
            pred = predict_words_from_frames(test_data, till)
        
            try:
                prediction = mode(pred)
            except:
                prediction = ''
            
            word.append(prediction)
        
        gold_label = video_name[0:3]
        print("\nSelection of Frame is Done\n")
        print("\nPredicting alphabets from frames extracted.")
        for i in range(0,6):
            if i == 3:
                print("generating keypoint timeseries for the word from posenet.csv")
            print("-")
            time.sleep(1)
            
    
        print("\nTrue Value: " + video_name[0:3] + " Prediction: " + prediction)

        time.sleep(1)
        pred_array.append([prediction, gold_label])

    df = DataFrame (pred_array,columns=['pred','true'])
    print(classification_report(df.pred, df.true))
    df.to_csv(join(video_path, 'results.csv'))
