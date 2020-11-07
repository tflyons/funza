package funza

import (
	gg "gorgonia.org/gorgonia"
)

type Option func(*modeler) error

type modeler struct {
	iter    int
	solver  gg.Solver
	useLisp bool
}
