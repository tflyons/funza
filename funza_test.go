package funza

import (
	"fmt"
	"testing"
)

func TestNewLogRegression(t *testing.T) {
	// extract data from csv
	xs, y, err := ReadCSV("study.csv", []string{"Hours"}, "Pass")
	if err != nil {
		t.Fatal(err)
	}

	// find thetas
	results, err := NewLogRegression(xs, y)
	if err != nil {
		t.Fatal(err)
	}

	// print
	for i := range results {
		fmt.Printf("theta_%d: %0.3f\n", i, results[i])
	}
}

func TestNewLinearRegression(t *testing.T) {
	// extract data from csv
	xs, y, err := ReadCSV("study.csv", []string{"Hours"}, "Pass")
	if err != nil {
		t.Fatal(err)
	}

	// find thetas
	results, err := NewLinearRegression(xs, y)
	if err != nil {
		t.Fatal(err)
	}

	// print
	for i := range results {
		fmt.Printf("theta_%d: %0.3f\n", i, results[i])
	}
}
