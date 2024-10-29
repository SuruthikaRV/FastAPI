import os
import uvicorn
import warnings
from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import pandas as pd
import tensorflow as tf

app = FastAPI()
warnings.filterwarnings("ignore")

# Load the pre-trained LSTM model
lstm_model = tf.keras.models.load_model('lstm_model.h5')

# Define the input data model based on the top 20 important features
class TrafficFeatures(BaseModel):
    packet_length_max: float = Field(..., example=-0.026769)
    fwd_iat_mean: float = Field(..., example=-0.467634)
    packet_length_mean: float = Field(..., example=0.042084)
    flow_iat_mean: float = Field(..., example=-0.471674)
    fwd_packets_length_total: float = Field(..., example=-0.212726)
    ack_flag_count: float = Field(..., example=-0.462813)
    avg_packet_size: float = Field(..., example=0.010255)
    subflow_bwd_packets: float = Field(..., example=-0.043853)
    subflow_bwd_bytes: float = Field(..., example=-0.015346)
    bwd_packet_length_max: float = Field(..., example=-0.162178)
    bwd_packets_length_total: float = Field(..., example=0.0)
    avg_fwd_segment_size: float = Field(..., example=-0.206625)
    init_fwd_win_bytes: float = Field(..., example=-0.206625)
    bwd_packet_length_mean: float = Field(..., example=-0.323819)
    avg_bwd_segment_size: float = Field(..., example=0.097813)
    urg_flag_count: float = Field(..., example=-0.384809)
    packet_length_min: float = Field(..., example=-0.054061)
    down_up_ratio: float = Field(..., example=0.096153)
    bwd_packets_per_s: float = Field(..., example=0.0)
    fwd_packet_length_min: float = Field(..., example=0.0)

@app.post("/predict")
async def predict(features: TrafficFeatures = Body(...)):
    # Prepare input data as a DataFrame and reshape it for the LSTM model
    input_data = pd.DataFrame({
        'Packet_Length_Max': [features.packet_length_max],
        'Fwd_IAT_Mean': [features.fwd_iat_mean],
        'Packet_Length_Mean': [features.packet_length_mean],
        'Flow_IAT_Mean': [features.flow_iat_mean],
        'Fwd_Packets_Length_Total': [features.fwd_packets_length_total],
        'ACK_Flag_Count': [features.ack_flag_count],
        'Avg_Packet_Size': [features.avg_packet_size],
        'Subflow_Bwd_Packets': [features.subflow_bwd_packets],
        'Subflow_Bwd_Bytes': [features.subflow_bwd_bytes],
        'Bwd_Packet_Length_Max': [features.bwd_packet_length_max],
        'Bwd_Packets_Length_Total': [features.bwd_packets_length_total],
        'Avg_Fwd_Segment_Size': [features.avg_fwd_segment_size],
        'Init_Fwd_Win_Bytes': [features.init_fwd_win_bytes],
        'Bwd_Packet_Length_Mean': [features.bwd_packet_length_mean],
        'Avg_Bwd_Segment_Size': [features.avg_bwd_segment_size],
        'URG_Flag_Count': [features.urg_flag_count],
        'Packet_Length_Min': [features.packet_length_min],
        'Down_Up_Ratio': [features.down_up_ratio],
        'Bwd_Packets_per_s': [features.bwd_packets_per_s],
        'Fwd_Packet_Length_Min': [features.fwd_packet_length_min]
    })

    # Reshape input data to fit LSTM model input requirements (e.g., 3D array: samples, timesteps, features)
    input_data = input_data.values.reshape((1, 1, input_data.shape[1]))

    # Predict using the loaded LSTM model
    prediction = lstm_model.predict(input_data)

    # Interpret the prediction
    if prediction[0][0] < 0.5:  # Adjust threshold as necessary based on model output
        return {"The prediction is an attack (0)"}
    else:
        return {"The prediction is benign (1)"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
