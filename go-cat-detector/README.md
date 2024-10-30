# Go cat-detector

Upload a photo and prompt [Google's Gemini API](https://ai.google.dev/gemini-api/docs) to determine if a tabby cat is present.

  - [Setup Gemeni API Key](#setup-gemeni-api-key)
  - [Testing](#testing)
  - [Running Server](#running-server)
  - [curl Request Examples](#curl-request-examples)


## Setup Postgres

Use [../docker-compose.yaml](../docker-compose.yaml) to bring up the pgvector dependency:
```
cd ../
docker-compose up
```

## Setup Gemeni API Key

 - Follow the Google docs to setup a [Gemini API-Key](https://ai.google.dev/gemini-api/docs/api-key)
 - Store the api-key in local keychain or secret store.

For instance, add GEMINI_API_KEY to OSX 10.13 keychain secret store:
```
security add-generic-password -a GEMINI_API_KEY -s cat-detector -w
password data for new item: <paste in key
retype password for new item:
```

 - Setup `.env`:
`cat .env`
```
#!/usr/bin/env bash
export GEMINI_API_KEY="$(security find-generic-password -a GEMINI_API_KEY -s cat-detector -w)
export GEMINI_API_ENDPOINT="https://generativelanguage.googleapis.com/upload/v1beta/files"
export DATABASE_URL="postgres://$(security find-generic-password -a DATABASE_USER_PASS  -s cat-detector -w)@localhost:5432/detector?sslmode=disable"
```


## Testing

`go test -v -cover -coverprofile cover.out -race ./...`:
```
Bryants-MBP:go-cat-detector bryant$ go test -v -cover -coverprofile cover.out .
# cat-detector.test
2024/10/29 22:08:45 github.com/testcontainers/testcontainers-go - Connected to docker:
  Server Version: 27.2.0
  API Version: 1.46
  Operating System: Docker Desktop
  Total Memory: 3914 MB
  Labels:
    com.docker.desktop.address=unix:///Users/bryant/Library/Containers/com.docker.docker/Data/docker-cli.sock
  Testcontainers for Go Version: v0.33.0
  Resolved Docker Host: unix:///Users/bryant/.docker/run/docker.sock
  Resolved Docker Socket Path: /var/run/docker.sock
  Test SessionID: ffdb49f1dac3fd680412d54040414dbf214c1ae2d719d1316c8523ca79d2c193
  Test ProcessID: 4d913cde-5378-4c4f-af7e-6d47be028607
2024/10/29 22:08:45 🐳 Creating container for image testcontainers/ryuk:0.8.1
2024/10/29 22:08:45 ✅ Container created: 3acf4949c1bf
2024/10/29 22:08:45 🐳 Starting container: 3acf4949c1bf
2024/10/29 22:08:45 ✅ Container started: 3acf4949c1bf
2024/10/29 22:08:45 ⏳ Waiting for container id 3acf4949c1bf image: testcontainers/ryuk:0.8.1. Waiting for: &{Port:8080/tcp timeout:<nil> PollInterval:100ms skipInternalCheck:false}
2024/10/29 22:08:46 🔔 Container is ready: 3acf4949c1bf
2024/10/29 22:08:46 🐳 Creating container for image pgvector/pgvector:pg16
2024/10/29 22:08:46 ✅ Container created: 78d14761e20e
2024/10/29 22:08:46 🐳 Starting container: 78d14761e20e
2024/10/29 22:08:46 ✅ Container started: 78d14761e20e
2024/10/29 22:08:46 ⏳ Waiting for container id 78d14761e20e image: pgvector/pgvector:pg16. Waiting for: &{Port:5432/tcp timeout:<nil> PollInterval:100ms skipInternalCheck:false}
2024/10/29 22:08:47 🔔 Container is ready: 78d14761e20e
=== RUN   TestAddImageHandler
[GIN-debug] [WARNING] Creating an Engine instance with the Logger and Recovery middleware already attached.

[GIN-debug] [WARNING] Running in "debug" mode. Switch to "release" mode in production.
 - using env:	export GIN_MODE=release
 - using code:	gin.SetMode(gin.ReleaseMode)

[GIN-debug] POST   /add_image                --> cat-detector.TestAddImageHandler.func1 (3 handlers)
[GIN] 2024/10/29 - 22:08:47 | 200 |    6.344056ms |                 | POST     "/add_image?imagePath=/Users/bryant/git/github.com/brybry192/cat-detector/images/beastie.jpg"
--- PASS: TestAddImageHandler (0.01s)
PASS
coverage: 21.2% of statements
2024/10/29 22:08:47 🐳 Terminating container: 78d14761e20e
2024/10/29 22:08:48 🚫 Container terminated: 78d14761e20e
ok  	cat-detector	2.400s	coverage: 21.2% of statements
```

## Running Server

Setup config and start go http server:
```
Bryants-MBP:go-cat-detector bryant$ source .env
Bryants-MBP:go-cat-detector bryant$ go run .
[GIN-debug] [WARNING] Creating an Engine instance with the Logger and Recovery middleware already attached.
[GIN-debug] [WARNING] Running in "debug" mode. Switch to "release" mode in production.
 - using env:	export GIN_MODE=release
 - using code:	gin.SetMode(gin.ReleaseMode)

[GIN-debug] POST   /add_image                --> main.main.func1 (3 handlers)
[GIN-debug] [WARNING] You trusted all proxies, this is NOT safe. We recommend you to set a value.
Please check https://pkg.go.dev/github.com/gin-gonic/gin#readme-don-t-trust-all-proxies for details.
[GIN-debug] Listening and serving HTTP on :8080
[GIN] 2024/10/29 - 22:00:34 | 200 |  5.119560354s |             ::1 | POST     "/add_image"
```

## curl Request Examples

Use a POST with the imagePath parameter to signal the location of image to feed to Gemini:
```
Bryants-MBP:go-cat-detector bryant$ curl -s -X POST http://localhost:8080/add_image?imagePath=/Users/bryant/git/github.com/brybry192/cat-detector/images/Beach_IMG_0699.jpg
{
  "image_id": 3,
  "is_beastie": false,
  "message":  "No, there is no tabby cat in this image. There are rocks and water.  \n",
  "uploaded_at": "2024-10-29T21:24:55.735615-07:00"
}
```

Sending an image with tabby cat present:
```
Bryants-MBP:go-cat-detector bryant$ curl -s -X POST http://localhost:8080/add_image?imagePath=/Users/bryant/git/github.com/brybry192/cat-detector/images/IMG_0366_beastie.jpg | jq .
{
  "err": null,
  "image_id": 5,
  "is_beastie": true,
  "message": " Yes, there is a tabby cat in the image. It is an orange tabby cat with white on its chest and belly. It is looking down at the floor and its tail is out of frame. The cat is in focus and the background is blurred.\n",
  "uploaded_at": "2024-10-29T21:48:32.15276-07:00"
}
```

Use a form to Uploading an image without tabby, but another cat:
```
Bryants-MBP:go-cat-detector bryant$ curl -s  -X POST http://localhost:8080/add_image -F image=@../images/IMG_3616_mac.jpg  | jq .
{
  "err": null,
  "image_id": 8,
  "is_beastie": false,
  "message": "No, the cat in the image is black.  It's hard to tell from the photo, but it looks like it may be a black shorthair. \n\n",
  "uploaded_at": "2024-10-29T22:00:34.897613-07:00"
}
```
