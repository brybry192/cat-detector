package main

import (
	"bytes"
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/jmoiron/sqlx"
	"github.com/stretchr/testify/assert"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

var (
	dbContainer testcontainers.Container
)

func TestMain(m *testing.M) {
	// Set up the PostgreSQL container
	ctx := context.Background()
	req := testcontainers.ContainerRequest{
		Image:        "pgvector/pgvector:pg16",
		ExposedPorts: []string{"5432/tcp"},
		Env: map[string]string{
			"POSTGRES_PASSWORD": "password",
			"POSTGRES_DB":       "image_db",
			"POSTGRES_USER":     "postgres",
		},
		WaitingFor: wait.ForListeningPort("5432/tcp"),
	}

	var err error
	dbContainer, err = testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		panic(err)
	}

	// Get the host and port of the container
	host, err := dbContainer.Host(ctx)
	if err != nil {
		panic(err)
	}

	port, err := dbContainer.MappedPort(ctx, "5432")
	if err != nil {
		panic(err)
	}

	// Build the connection string
	dsn := "postgres://postgres:password@" + host + ":" + port.Port() + "/image_db?sslmode=disable"

	// Connect to the database
	db, err = sqlx.Connect("postgres", dsn)
	if err != nil {
		panic(err)
	}

	// Initialize the database schema
	InitSchema(db)

	// Run tests
	code := m.Run()

	// Teardown
	if err := dbContainer.Terminate(ctx); err != nil {
		panic(err)
	}

	os.Exit(code)
}

func TestAddImageHandler(t *testing.T) {
	// Initialize services
	// Use mock embedding service
	embeddingService := &MockEmbeddingService{}

	// Create a test router with dependency injection
	router := gin.Default()
	router.POST("/add_image", func(c *gin.Context) {
		AddImageHandler(c, db, embeddingService)
	})

	// Create a test image file (Use a small dummy image)
	imagePath := "../images/beastie.jpg"
	imageData, err := os.ReadFile(imagePath)
	assert.NoError(t, err)

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("image", "beastie.jpg")
	assert.NoError(t, err)

	_, err = part.Write(imageData)
	assert.NoError(t, err)
	writer.Close()

	req, err := http.NewRequest("POST", "/add_image?imagePath=/Users/bryant/git/github.com/brybry192/cat-detector/images/beastie.jpg", body)
	assert.NoError(t, err)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Perform the request
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	// Check the response
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "Yes. A tabby cat was found.")
}

type MockEmbeddingService struct{}

func (m *MockEmbeddingService) GetImageEmbedding(imageBytes []byte) ([]float64, error) {
	// Return a fixed embedding vector for testing
	embedding := make([]float64, 1696)
	for i := range embedding {
		embedding[i] = 0.5 // or any other dummy value
	}
	return embedding, nil
}
func (m *MockEmbeddingService) UploadImageAndPrompt(imageBytes []byte, imagePath, prompt string) (string, error) {
	return "Yes. A tabby cat was found.", nil
}
