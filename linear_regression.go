package funza

import (
	"fmt"

	"github.com/pkg/errors"

	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

//NewLinearRegression returns the scalar θ values for multivariable linear regression
//  Y = θ0 + θ1*x1 + ... + θN*xN
func NewLinearRegression(xValues [][]float64, yValues []float64, opts ...Option) ([]float64, error) {
	m := &modeler{
		iter:    10000,
		solver:  gg.NewAdamSolver(gg.WithLearnRate(0.1)),
		useLisp: false,
	}

	// replace defaults with options
	for _, opt := range opts {
		err := opt(m)
		if err != nil {
			return nil, errors.Wrap(err, "invalid option")
		}
	}

	// graph
	graph := gg.NewGraph()

	// turn data into vector node
	Y := gg.NewVector(graph, gg.Float64, gg.WithName("y"), gg.WithShape(len(yValues)), gg.WithValue(
		tensor.New(tensor.WithBacking(yValues), tensor.WithShape(len(yValues))),
	))

	// get scalars for calculation
	var thetas []*gg.Node

	// theta0 + theta1*X1 + theta2*X2
	thetas = append(thetas, gg.NewScalar(graph, gg.Float64, gg.WithValue(1.0), gg.WithName("theta_0")))
	p := thetas[0]
	for i, x := range xValues {
		// vector node
		X := gg.NewVector(graph, gg.Float64, gg.WithName(fmt.Sprintf("x_%d", i+1)), gg.WithShape(len(x)), gg.WithValue(
			tensor.New(tensor.WithBacking(x), tensor.WithShape(len(x))),
		))

		// scalar node
		theta := gg.NewScalar(graph, gg.Float64, gg.WithValue(1.0), gg.WithName(fmt.Sprintf("theta_%d", i+1)))
		thetas = append(thetas, theta)

		// thetaN * xN
		mul, err := gg.Mul(X, theta)
		if err != nil {
			return nil, errors.Wrap(err, "x*theta multiplication error")
		}

		// p += mul
		p, err = gg.Add(p, mul)
		if err != nil {
			return nil, errors.Wrap(err, "x*theta addition error")
		}
	}

	// solve
	return solve(graph, m, thetas, p, Y)
}
