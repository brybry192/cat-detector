package main

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/jmoiron/sqlx"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

var db *sqlx.DB

func InitDB() *sqlx.DB {
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found")
	}

	databaseURL := os.Getenv("DATABASE_URL")
	var errDB error
	db, errDB = sqlx.Connect("postgres", databaseURL)
	if errDB != nil {
		log.Fatalln(errDB)
	}

	return db
}

// Float64Array is a custom type for []float64
type Float64Array []float64

// Value implements the driver.Valuer interface
func (a Float64Array) Value() (driver.Value, error) {
	strs := make([]string, len(a))
	for i, v := range a {
		strs[i] = fmt.Sprintf("%f", v)
	}
	return "[" + strings.Join(strs, ",") + "]", nil
}

// Scan implements the sql.Scanner interface
func (a *Float64Array) Scan(src interface{}) error {
	if src == nil {
		*a = nil
		return nil
	}

	srcStr, ok := src.(string)
	if !ok {
		return errors.New("Float64Array: source is not a string")
	}

	srcStr = strings.Trim(srcStr, "{}")
	strValues := strings.Split(srcStr, ",")
	floatValues := make([]float64, len(strValues))
	for i, str := range strValues {
		var val float64
		_, err := fmt.Sscanf(str, "%f", &val)
		if err != nil {
			return err
		}
		floatValues[i] = val
	}

	*a = floatValues
	return nil
}

func InitSchema(db *sqlx.DB) {
	schema := `
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS images (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        embedding VECTOR(1696),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	UNIQUE (filename)
    );`

	db.MustExec(schema)
}
