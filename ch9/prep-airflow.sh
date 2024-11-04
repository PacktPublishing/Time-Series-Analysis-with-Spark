mkdir -p ./dags ./logs ./plugins ./config
echo "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
#echo "AIRFLOW_UID=$(id -u)" > .env
