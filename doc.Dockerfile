FROM nginx:latest
COPY docs/_build/html/ /usr/share/nginx/html/