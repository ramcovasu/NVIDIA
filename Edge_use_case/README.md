# NVIDIA-Gen-AI

# This is a project created for NVIDIA Gen AI submission.

# My Use Case: How do you make Edge Devices to be self reliant using LLM's

# Use case has multiple compoments : 1. Component 1 - Synthetic data generation for Devies. 2. Crew AI Agent Using Llama 3 70B detecting Anomalies in the device data which has device id, temperature, pressure, voltage, co2, rotation_speed etc. 3. A RAG is built using FAISS index to store details about the actions to take for different anomalies 4. An open source Gorilla LLM used specifically for function calling and ability to take specific actions like send_email etc 5. A streamlit app just for final output and for uploading the known issues & resolution guide to build the knowledge graph (FAISS index). 
