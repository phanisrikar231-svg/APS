# FedMSME-PdM

Python-only software demo for a BS capstone project:

**FedMSME-PdM: An Online Federated Predictive Maintenance Dashboard for Indian MSMEs Using Simulated Industrial Sensor Data**

The project demonstrates the software intelligence layer of predictive maintenance. Real shop-floor hardware is represented by benchmark-style/simulated industrial CSV sensor streams, so the demo can run fully online on a normal PC.

## Saved Location

```text
C:\Users\phani\Documents\Codex\2026-05-17\hey-i-have-a-project-domaun
```

## Features

- Signup and login system
- Six-digit OTP verification with expiry and attempt limits
- Demo OTP display when SMTP is not configured
- MSME and machine registration
- Expanded heterogeneous machine support: CNC, lathe, drilling, textile motor, injection molding, hydraulic press, pump, compressor, conveyor motor, packaging machine, cooling fan, oven, welding transformer, boiler feed pump, gearbox, general motor, and custom machines
- CSV sensor upload/paste workflow
- Built-in Safe / Warning / Critical demo CSV generator
- Common machine-health feature extraction for different sensor schemas
- Federated learning simulation using FedAvg and FedProx
- Failure risk, health score, and Remaining Useful Life prediction
- Exact critical-area diagnosis such as bearing/shaft vibration, thermal/cooling, electrical load, pressure, acoustic, speed/drive, and mechanical load zones
- Local machine chatbot that answers from the machine profile, latest prediction, sensor batch, and explanation data
- Public dataset guidance page
- PDF machine-health report download
- Seeded demo account and sample CSV files

## Demo Login

```text
Email: admin@demo.msme
Password: demo1234
```

## Run

Use Python 3.10+ with the libraries in `requirements.txt`.

```powershell
cd C:\Users\phani\Documents\Codex\2026-05-17\hey-i-have-a-project-domaun
python -m pip install -r requirements.txt
python run_app.py --port 8000
```

In this Codex workspace, the bundled Python command is:

```powershell
& 'C:\Users\phani\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_app.py --port 8000
```

Or run the included PowerShell helper:

```powershell
cd C:\Users\phani\Documents\Codex\2026-05-17\hey-i-have-a-project-domaun
.\start_app.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## How To Present Without Hardware

Use this explanation:

> This project focuses on the software intelligence layer of predictive maintenance. Real machines would provide IoT/PLC sensor streams in deployment, but the capstone demo uses simulated industrial sensor CSVs to represent machine data. The web app demonstrates user verification, machine onboarding, federated training, failure prediction, RUL estimation, explainable alerts, and report generation.

## How To Use Without CSV Files

Open a machine page and click one of:

```text
Generate Safe CSV
Generate Warning CSV
Generate Critical CSV
```

The app creates a realistic sensor batch for that machine type and runs prediction immediately.

You can also use public datasets:

- NASA PCoE Data Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- UCI AI4I 2020 Predictive Maintenance Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter

## Machine Chatbot

The chatbot is a local lightweight context bot. It does not call an external API. It answers using only:

- the selected machine profile,
- latest sensor batch,
- latest prediction,
- critical-area explanation,
- RUL and health score.

Ask questions like:

```text
Why is this machine critical?
What should the technician inspect?
What sensors are used?
What is the remaining useful life?
How does federated learning help privacy?
```

## Why Different MSMEs Can Still Train Together

Different MSMEs may use different machines and workflows, so their data is non-IID. The project handles this by:

- converting raw sensor streams into a common health-feature vector,
- including sensor-availability flags,
- simulating separate clients for each MSME,
- comparing FedAvg and FedProx,
- storing raw CSV data locally instead of sending it to a central server.

The strongest research angle is:

> Privacy-preserving federated predictive maintenance for heterogeneous Indian MSME machine environments using common health-feature extraction and FedProx personalization.

## Project Structure

```text
fedmsme/
  app.py          Web routes and dashboard pages
  database.py     SQLite schema
  ml.py           Feature extraction, FedAvg/FedProx, prediction
  reports.py      PDF report generation
  security.py     Password hashing, sessions, OTP
  seed.py         Demo account, machines, sample data
  templates.py    HTML rendering helpers
static/
  style.css       Dashboard styling
sample_data/      Generated demo CSV files after first run
run_app.py        App entrypoint
```
