## Building docker image on host

```sh
cd /var/www/classification-predict-streamlit-template
```

```bash
sudo docker build -t streamlit .
```

## Running docker image 

### ! Dont run this command if it has already been run. waste of time!

```bash
sudo docker run --name streamlit_1 -d -p 8501:8501 streamlit
```

# RUN THIS CODE ON WEDNESDAY 

```bash
sudo docker start streamlit_1 
```

## App will be running on:

```
http://<your instance IP>:8501
```