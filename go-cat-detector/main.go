// main.go
package main

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {

	// to change the flags on the default logger
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Initialize services
	db := InitDB()
	InitSchema(db)

	// Load environment variables
	geminiAPIKey := os.Getenv("GEMINI_API_KEY")
	// Initialize the embedding service
	embeddingService, err := NewGeminiEmbeddingService(geminiAPIKey)
	if err != nil {
		log.Fatalf("Failed to initialize embedding service: %v", err)
	}

	defer embeddingService.Client.Close()

	// Set up router
	router := gin.Default()
	router.POST("/add_image", func(c *gin.Context) {
		AddImageHandler(c, db, embeddingService)
	})

	// Start server
	router.Run(":8080")
}
