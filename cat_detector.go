package main

import (
	"flag"
	"fmt"
	"image"
	"image/draw"
	"log"
	"os"
	"path/filepath"
	"strings"

	yolov3 "github.com/wimspaargaren/yolov3"
	"gocv.io/x/gocv"
)

// CropImage crops the image to the bounding box specified by (x, y, w, h)
func CropImage(img image.Image, box image.Rectangle) image.Image {
	// Calculate cropping coordinates
	rect := image.Rect(int(box.Min.X), int(box.Min.Y), int(box.Max.X), int(box.Max.Y))

	// Create a new blank image to hold the cropped region
	croppedImg := image.NewRGBA(rect.Bounds())

	// Draw the cropped part of the original image onto the new blank image
	draw.Draw(croppedImg, croppedImg.Bounds(), img, rect.Min, draw.Src)

	return croppedImg
}

func main() {
	// Set up a flag to pass the directory, leaving Go to handle the glob pattern
	imgPath := flag.String("i", "images", "Image path or directory containing jpg images")
	flag.Parse()

	// Initialize the model
	yolo, err := yolov3.NewNet("yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3/coco.names")
	if err != nil {
		log.Fatal(err)
	}
	defer yolo.Close()

	// Gracefully close the net when the program is done
	defer func() {
		err := yolo.Close()
		if err != nil {
			log.Printf("unable to gracefully close yolo net: %v", err)
		}
	}()

	// Check if the path is a directory or file.
	statInfo, err := os.Stat(*imgPath)
	if err != nil {
		log.Fatalf("Error accessing path: %v", err)
	}

	var files []string
	if statInfo.IsDir() {
		// If it's a directory, use a glob to find .jpg files
		pattern := filepath.Join(*imgPath, "*.jpg")
		files, err = filepath.Glob(pattern)
		if err != nil {
			log.Fatalf("Error finding .jpg files: %v", err)
		}
	}

	if !statInfo.IsDir() {
		files = append(files, *imgPath)
	}

	// Loop over each file and process it
	for _, file := range files {
		// Get the base name of the file
		//fileName := filepath.Base(file)
		fmt.Println("Processing:", file)
		frame := gocv.IMRead(file, gocv.IMReadColor)
		//img, err := frame.ToImage()
		//if err != nil {
		//	log.Fatal(err)
		//}

		// Perform detection
		detections, err := yolo.GetDetections(frame)
		if err != nil {
			log.Fatal(err)
		}
		for i, detection := range detections {
			if detection.ClassName == "cat" {
				fmt.Printf("Box %v detected class ID %v for %v at %v with confidence %v\n", i+1, detection.ClassID, detection.ClassName, detection.BoundingBox, detection.Confidence)
				//croppedCat := CropImage(img, detection.BoundingBox)

				// Save the cropped image to a file
				//outFile, err := os.Create(detectedImg)
				//if err != nil {
				//	fmt.Println("Error saving cropped image:", err)
				//	continue
				//}
				//defer outFile.Close()

				//jpeg.Encode(outFile, croppedCat, nil)
				//fmt.Printf("Cropped cat saved as cat_%d.jpg\n", i)

				//// Load the image
				//imgCat := gocv.IMRead(fmt.Sprintf("cat_%d.jpg", i), gocv.IMReadColor)
				//if imgCat.Empty() {
				//	fmt.Println("Failed to load image:", fmt.Sprintf("cat_%d.jpg", i))
				//	continue
				//}
				//defer imgCat.Close()

				// Preprocess the image
				//gocv.Resize(imgCat, &imgCat, image.Pt(224, 224), 0, 0, gocv.InterpolationLinear)
				//gocv.Normalize(imgCat, &imgCat, 0, 1, gocv.NormMinMax)

				//// Load the pre-trained model (ResNet-50 in this case)
				//net := gocv.ReadNet("/Users/bryant/models/yolov3/yolov3.onnx", "")
				//if net.Empty() {
				//	fmt.Println("Failed to load the network")
				//	return
				//}
				//defer net.Close()

				//// Convert image to blob for inference
				//blob := gocv.BlobFromImage(imgCat, 1.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), true, false)
				//defer blob.Close()

				//// Set the input blob for the network
				//net.SetInput(blob, "data")

				//// Run inference
				//embedding := net.Forward("fc1000")
				//defer embedding.Close()

				// Print the embedding vector
				//fmt.Println("Embedding Vector:", embedding)

			}
		}

		// Draw bounding box around detections.
		yolov3.DrawDetections(&frame, detections)

		// Save the modified image with bounding boxes.
		detectedImg := strings.ReplaceAll(file, ".jpg", "_detected.jpg")
		ok := gocv.IMWrite(detectedImg, frame)
		if !ok {
			log.Fatalf("Error: Could not save image %v\n", file)
		}
	}

	//window := gocv.NewWindow("Result Window")
	//defer func() {
	//	err := window.Close()
	//	if err != nil {
	//		log.Println("unable to close window")
	//	}
	//}()

	//window.IMShow(frame)
	//window.ResizeWindow(872, 585)

	//window.WaitKey(10000000000)
}
