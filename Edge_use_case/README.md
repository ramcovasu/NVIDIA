# NVIDIA-Gen-AI

# This is a project created for NVIDIA Gen AI submission.

# My Use Case: How do you make Edge Devices (IOT) to be self reliant using LLM's. An anomaly occurs in the device data, an agent detects the anomaly, searches the knowledge graph for resolution 3 Another agent generates the function call for fixing/taking action. Actions could be sending emails or restarting device etc. 

# Use case has multiple compoments : 
Component 1 - Synthetic data generation for IOT Devices.
2. Crew AI Agent Using Llama 3 70B for detecting Anomalies in the device data which has device id, temperature, pressure, voltage, co2, rotation_speed etc. As most of the smaller open source LLM's are not able to do Anomaly detection efficiently. In next release, i am trying to fine tune a Llama 3 7B LLM for Anomaly detection. I have used Gorq to run the Llama 3 70B model.
3. A RAG is built using FAISS index to store details about the actions to take for different anomalies
4. An open source Gorilla LLM used specifically for function calling and ability to take specific actions like send_email etc
5. A streamlit app just for final output and for uploading the known issues & resolution guide to build the knowledge graph (FAISS index). 
