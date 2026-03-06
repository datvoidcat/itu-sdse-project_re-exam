// Dagger pipeline for MLOps project
// This program uses Dagger to run ML training and testing in containers

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"dagger.io/dagger"
)

const (
	pythonImg = "python:3.11-slim"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go [train|test]")
		os.Exit(2)
	}

	cmd := os.Args[1]
	ctx := context.Background()

	// Connect to Dagger engine for running containers
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		log.Fatalf("failed to connect to dagger: %v", err)
	}
	defer client.Close()

	// Get the repository source directory
	source := client.Host().Directory("..")

	// Run the appropriate command based on user input
	case "test":
		if err := runTest(ctx, client, source); err != nil {
			log.Fatal(err)
		}
	case "train":
		if err := runTrain(ctx, client, source); err != nil {
			log.Fatal(err)
		}
	default:
		fmt.Println("Unknown command:", cmd)
		fmt.Println("Usage: go run main.go [test|train|predict]")
		os.Exit(2)
	}
}

func basePythonContainer(client *dagger.Client, source *dagger.Directory) *dagger.Container {
	// Create a base Python container with dependencies installed
	c := client.Container().
		From(pythonImg).
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "--no-cache-dir", "-r", "requirements.txt"})

	return c
}

func runTrain(ctx context.Context, client *dagger.Client, source *dagger.Directory) error {
	// Run the training pipeline in a container
	c := basePythonContainer(client, source)

	// Pull data from DVC
	c = c.WithExec([]string{"sh", "-c", "dvc pull || true"})

	// Run the training pipeline
	c = c.WithWorkdir("/workspace/MLOps_Project").
		WithExec([]string{"python", "-m", "pipeline"})

	// Export the trained model to the host
	model := c.File("/workspace/models/model.pkl")
	_, err := model.Export(ctx, "../models/model.pkl")
	return err
}

func runTest(ctx context.Context, client *dagger.Client, source *dagger.Directory) error {
	// Run inference tests in a container
	c := client.Container().
		From("python:3.11-slim").
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "--no-cache-dir", "-r", "requirements.txt"})

	// Debug: show what's present in the container
	c = c.WithExec([]string{"sh", "-c", "ls -lah /workspace || true"})
	c = c.WithExec([]string{"sh", "-c", "ls -lah /workspace/models || true"})
	c = c.WithExec([]string{"sh", "-c", "ls -lah /workspace/data/processed/artifacts || true"})

	// Run inference test (print full traceback, fail on error)
	c = c.WithExec([]string{"sh", "-c", "set -euo pipefail; python -u tests/model_inference.py"})

	_, err := c.Sync(ctx)
	return err
}
