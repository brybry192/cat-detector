package main

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jmoiron/sqlx"
)

func AddImageHandler(c *gin.Context, db *sqlx.DB, embeddingService EmbeddingService) {
	// Access query parameters directly using Gin context

	// If you need to get a specific query parameter
	imagePath := c.Query("imagePath")
	imageData := []byte{}

	var err error
	// Get the image from value.
	if len(imagePath) > 0 {

		imageData, err = os.ReadFile(imagePath)
		if err != nil {
			fmt.Println(gin.H{"error": err})
			c.JSON(http.StatusBadRequest, gin.H{"error": err})
			return
		}
	} else {

		// Parse the uploaded file using Gin's context method
		header, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No image uploaded"})
			return
		}

		// Open the uploaded file
		file, err := header.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Unable to open uploaded file"})
			return
		}
		defer file.Close()

		// Read the image data
		imageData, err = io.ReadAll(file)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Unable to read image data"})
			return
		}
		imagePath = header.Filename
	}

	// Get the embedding using the EmbeddingService`
	resp, err := embeddingService.UploadImageAndPrompt(imageData, imagePath, "Do you see a tabby cat in this image?")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to check image"})
		return
	}

	// TODO: Generate a real embedding
	embedding, err := GetImageEmbedding(1696)
	if err != nil {
		fmt.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image embedding"})
		return
	}

	// Save embedding to the database
	image := Image{
		Filename:  imagePath,
		Embedding: embedding,
		CreatedAt: time.Now(),
	}

	query := `INSERT INTO images (filename, embedding, created_at) VALUES ($1, $2, $3) RETURNING id`
	err = db.QueryRow(query, image.Filename, Float64Array(embedding), image.CreatedAt).Scan(&image.ID)
	if err != nil {
		fmt.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image embedding"})
		return
	}

	// Check resp string for confirmation of tabby.
	isBeastie := false
	if strings.Contains(resp, "Yes") {
		isBeastie = true
	}

	c.JSON(http.StatusOK, gin.H{
		"message":     resp,
		"image_id":    image.ID,
		"is_beastie":  isBeastie,
		"err":         nil,
		"uploaded_at": image.CreatedAt,
	})
}

func CosineSimilarity(a, b []float64) float64 {
	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func GetImageEmbedding(size int) ([]float64, error) {
	// Return a fixed embedding vector for testing
	embedding := make([]float64, size)
	for i := range embedding {
		embedding[i] = 0.5 // or any other dummy value
	}
	return embedding, nil
}
