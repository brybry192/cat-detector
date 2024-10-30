// database_test.go
package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFloat64Array_Value(t *testing.T) {
	arr := Float64Array{1.23, 4.56, 7.89}
	val, err := arr.Value()
	assert.NoError(t, err)
	assert.Equal(t, "[1.230000,4.560000,7.890000]", val)
}

func TestFloat64Array_Scan(t *testing.T) {
	var arr Float64Array
	err := arr.Scan("{1.23,4.56,7.89}")
	assert.NoError(t, err)
	assert.Equal(t, Float64Array{1.23, 4.56, 7.89}, arr)
}

func TestInitSchema(t *testing.T) {
	// Assuming db is already connected to the test database
	InitSchema(db)

	var tableName string
	err := db.QueryRow("SELECT table_name FROM information_schema.tables WHERE table_name='images'").Scan(&tableName)
	assert.NoError(t, err)
	assert.Equal(t, "images", tableName)
}
