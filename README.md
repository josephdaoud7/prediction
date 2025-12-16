<!-- create environment-->
python3 -m venv venv

<!-- activate environment -->
source venv/bin/activate

<!-- install packages -->
pip install -r requirements.txt

<!-- authenticate bigquery -->
gcloud auth application-default login