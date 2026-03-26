# Support Ticket Triage — OpenEnv 🚀
**Team:** Fresh Tensors (Vikash S, Sona R, Jositha)  
**Event:** Meta PyTorch OpenEnv Hackathon x Scaler School of Technology  
**Live API Endpoint:** [Hugging Face Space](https://huggingface.co/spaces/vikashsaravanan/openenv-support-triage)

## Project Abstract
This repository contains a production-ready OpenEnv simulation for training AI agents in frontline customer support triage. By interacting with this API, AI models learn to categorize tickets, assign priority levels, and route issues to specialized human teams—drastically reducing Time to Resolution (TTR) for growing startups.

## Technical Architecture
* **Framework:** Python, FastAPI
* **Deployment:** Docker, Hugging Face Spaces (Configured with non-root UID 1000 for cloud security)
* **Endpoints Supported:** `/reset`, `/step`, `/state`, `/tasks`, `/grader`, and `/health`.

## Task Difficulty Levels
1. **Easy:** Ticket Classification (Billing, Tech, Shipping, etc.)
2. **Medium:** Triage & Routing (Category + Priority + Assigned Team)
3. **Hard:** Full Resolution (Routing + Drafting a dynamic customer response)
