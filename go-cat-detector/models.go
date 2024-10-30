package main

import (
	"time"
)

type Image struct {
	ID        int       `db:"id"`
	Filename  string    `db:"filename"`
	Embedding []float64 `db:"embedding"`
	CreatedAt time.Time `db:"created_at"`
}
