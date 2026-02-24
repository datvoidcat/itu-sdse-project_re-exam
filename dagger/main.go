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
	if len(os.Args) < 1 {
		fmt.Println("Usage: go run main.go [train|test") // setup for predit and test as well
		os.Exit(2)
	}

	cmd := os.Args[1]
	ctx := context.Background()

	// Connect to Dagger engine (uses Docker Desktop under the hood)
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		log.Fatalf("failed to connect to dagger: %v", err)
	}
	defer client.Close()

	// Repository root is one level up from /dagger
	source := client.Host().Directory("..")

	switch cmd {
	case "test":
		if err := runTests(ctx, client, source); err != nil {
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
	c := client.Container().
		From(pythonImg).
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "--no-cache-dir", "-r", "requirements.txt"})

	return c
}

func pullDataIfPossible(c *dagger.Container) *dagger.Container {
	// We don’t want this to fail the entire pipeline if remote is flaky.
	return c.WithExec([]string{"sh", "-c", "dvc pull || true"})
}

func runTrain(ctx context.Context, client *dagger.Client, source *dagger.Directory) error {
	c := basePythonContainer(client, source)

	// Pull data
	c = c.WithExec([]string{"sh", "-c", "dvc pull || true"})

	// Run pipeline from inside MLOps_Project so `import config` works
	c = c.WithWorkdir("/workspace/MLOps_Project").
		WithExec([]string{"python", "-m", "pipeline"})

	// Export model from container -> host (repo root)
	model := c.File("/workspace/models/model.pkl")
	_, err := model.Export(ctx, "../models/model.pkl")
	return err
}

func runTests(ctx context.Context, client *dagger.Client, source *dagger.Directory) error {
	c := basePythonContainer(client, source)
	c = pullDataIfPossible(c)

	_, err := c.Sync(ctx)
	return err
}
