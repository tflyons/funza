package funza

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/pkg/errors"
)

func ReadCSV(filename string, xHeaders []string, yHeader string) ([][]float64, []float64, error) {
	// range of data
	var xs [][]float64
	var y []float64

	// read csv
	f, err := os.Open(filename)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "could not open file %s", filename)
	}
	defer f.Close()
	r := csv.NewReader(f)

	// read records
	indexByName := map[string]int{
		yHeader: -1,
	}
	for _, h := range xHeaders {
		indexByName[h] = -1
	}
	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, errors.Wrapf(err, "could not read file %s as csv", filename)
	}

	// get header indices
	for i, s := range records[0] {
		if s == yHeader {
			indexByName[s] = i
			continue
		}
		for _, h := range xHeaders {
			if s == h {
				indexByName[s] = i
				break
			}
		}
	}

	// check all headers found
	for h, i := range indexByName {
		if i == -1 {
			return nil, nil, errors.Errorf("could not find header: %s", h)
		}
	}

	// read values from csv
	xs = make([][]float64, len(xHeaders))
	for _, recs := range records[1:] {
		for i, h := range xHeaders {
			value, err := strconv.ParseFloat(recs[indexByName[h]], 64)
			if err != nil {
				return nil, nil, errors.Wrapf(err, "could not parse record as float %s", recs[indexByName[h]])
			}
			xs[i] = append(xs[i], value)
		}
		value, err := strconv.ParseFloat(recs[indexByName[yHeader]], 64)
		if err != nil {
			return nil, nil, errors.Wrapf(err, "could not parse record as float %s", recs[indexByName[yHeader]])
		}
		y = append(y, value)
	}

	return xs, y, nil
}
