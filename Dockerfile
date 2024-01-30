FROM python:3.9
# Or any preferred Python version.
ADD multiple_modal_net/main.py .

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./multiple_modal_net/main.py" ]