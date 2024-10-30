package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	// Import the GenerativeAI package for Go
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type EmbeddingService interface {
	UploadImageAndPrompt(imageBytes []byte, imagePath, prompt string) (string, error)
}

type GeminiEmbeddingService struct {
	Client *genai.Client
}

func NewGeminiEmbeddingService(apiKey string) (*GeminiEmbeddingService, error) {
	ctx := context.Background()

	// Access GEMINI_API_KEY as an environment variable.
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}
	return &GeminiEmbeddingService{
		Client: client,
	}, nil
}

func (g *GeminiEmbeddingService) UploadImageAndPrompt(image []byte, imagePath, prompt string) (string, error) {
	ctx := context.Background()

	// Access GEMINI_API_KEY as an environment variable.
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		return "", err
	}

	if len(image) > 0 {
		// Create a temporary file
		tmpFile, err := os.CreateTemp("", fmt.Sprintf("upload-%v.jpg", imagePath))
		if err != nil {
			return "", err
		}
		defer os.Remove(tmpFile.Name())

		// Write image bytes to the temporary file
		if _, err := tmpFile.Write(image); err != nil {
			tmpFile.Close()
			return "", err
		}
		tmpFile.Close()

		imagePath = tmpFile.Name()

	}

	file, err := client.UploadFileFromPath(ctx, filepath.Join(imagePath), nil)
	if err != nil {
		return "", err
	}
	defer client.DeleteFile(ctx, file.Name)

	model := client.GenerativeModel("gemini-1.5-flash")
	resp, err := model.GenerateContent(ctx,
		genai.FileData{URI: file.URI},
		genai.Text(prompt))
	if err != nil {
		return "", err
	}

	return getResponse(resp), nil
}

func getResponse(resp *genai.GenerateContentResponse) (r string) {
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				r = fmt.Sprintf("%v%v\n", r, part)
			}
		}
	}
	return r
}
