# 前提
Using deep learning to predict future temperature and humidity, and then calculate it into a **comfort index** and send it to the hardware that controls the temperature of the air-conditioning, so as to automatically adjust the temperature of the air-conditioning to a more comfortable temperature.

## What is confort index ?

The comfort index is an index that combines the comprehensive effects of temperature, humidity, and other meteorological elements on the human body to indicate whether the human body is comfortable in the atmospheric environment. 



![image](https://user-images.githubusercontent.com/58096503/232561098-6b6edfad-972e-4caf-bc4c-e55b260f4288.png)



# 【運作流程】

![image](https://user-images.githubusercontent.com/58096503/232563206-7fae8627-fa05-4d0f-9395-92d0bb1b0f32.png)

# 【使用硬體介紹】


![image](https://user-images.githubusercontent.com/58096503/232563046-74e1f282-a725-4679-a796-c5b304197e03.png)


# 【數據收集】
Using the Modbus RTU Communication Protocol to read the data from M12FT3Q sensor

![image](https://user-images.githubusercontent.com/58096503/232570258-35d6fff8-dfbc-42ce-a225-6aa73369568a.png)


# 【訓練和預測數據】

Before predicting the data must train the model

1. First the data feed into the LSTM network to train
2. Second the data feed into trained model to predict the dataTemperature
3. the result prediction include the DateTime, Temperature, and Humidity


![image](https://user-images.githubusercontent.com/58096503/232574935-5786376c-3fc3-4b25-ae72-057de30dd2d3.png)

# 【數據可視化】


![image](https://user-images.githubusercontent.com/58096503/232578173-e0ca0f54-0b86-4b78-8f3e-af9e9f5d50e2.png)






