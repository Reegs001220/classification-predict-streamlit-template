# Building docker image on host

```sh
cd /var/www/classification-predict-streamlit-template
```

## ! Dont run this command if it has already been run. waste of time!

```bash
sudo docker build -t streamlit .
```

# Running docker image 

```bash
sudo docker run --name streamlit_1 -d -p 8501:80 streamlit
```