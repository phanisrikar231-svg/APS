# Capstone Brief

## Title

**FedMSME-PdM: A Python-Based Online Federated Predictive Maintenance Dashboard for Heterogeneous Indian MSMEs**

## Problem Statement

Indian MSMEs often operate CNC machines, motors, pumps, compressors, and other production assets with limited maintenance digitization. Sudden failures cause downtime, repair cost, missed delivery deadlines, and quality losses. Individual MSMEs usually lack enough failure data to train accurate predictive models and may not be willing to share raw machine data with other companies.

This project builds a software-only federated predictive maintenance platform that allows multiple MSMEs to collaboratively train a machine-health model without sharing raw sensor CSVs.

## Abstract

FedMSME-PdM is a Python-based web dashboard for privacy-preserving predictive maintenance in Indian MSME environments. The system supports OTP-verified user onboarding, expanded machine registration, custom machine types, generated demo sensor CSVs, sensor CSV upload, federated training simulation, machine failure-risk prediction, Remaining Useful Life estimation, exact critical-area diagnosis, PDF reports, and a local machine chatbot. Since different MSMEs may use different machines and sensor schemas, raw sensor streams are transformed into a common health-feature vector containing vibration intensity, temperature trend, current fluctuation, pressure level, load, operating hours, and sensor availability flags. Federated learning is simulated with heterogeneous clients, comparing FedAvg and FedProx under non-IID machine conditions. The final dashboard demonstrates how a software-only predictive maintenance layer can be evaluated using benchmark-style simulated sensor data without physical hardware.

## What The App Solves

- predicts machine failure risk before breakdown,
- estimates Remaining Useful Life,
- lets MSMEs collaborate without sharing raw data,
- supports different machines and workflows,
- gives technicians exact critical zones behind warnings,
- lets users ask a local chatbot about their own machine condition,
- produces downloadable maintenance reports.

## Web App Features

- login, signup, and OTP verification,
- MSME demo profile,
- machine registration,
- sensor CSV upload/paste,
- Safe / Warning / Critical demo CSV generation,
- seeded sample data for demo factories,
- dashboard with health score, risk, RUL, and status,
- exact diagnosis zones: bearing, thermal, electrical, pressure, acoustic, speed, and load,
- local lightweight context chatbot,
- public dataset guidance page,
- federated training page,
- FedAvg and FedProx comparison,
- explanation of non-IID heterogeneous data handling,
- PDF report download.

## Technology

- Python 3.10+
- SQLite
- NumPy
- Pandas
- ReportLab
- Standard-library HTTP server
- No JavaScript runtime required

## Future Improvements

- real IoT/PLC data integration,
- production Django/FastAPI deployment,
- SMS OTP using a provider such as Twilio/Fast2SMS,
- differential privacy,
- secure aggregation,
- local model personalization per machine type,
- mobile technician interface,
- multilingual dashboard for MSME shop floors.
