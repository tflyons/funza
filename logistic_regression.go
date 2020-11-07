package funza

import (
	"fmt"
	"log"
	"runtime"

	"github.com/pkg/errors"

	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func NewLogRegression(xValues [][]float64, yValues []float64, opts ...Option) ([]float64, error) {
	m := &modeler{
		iter:    10000,
		solver:  gg.NewVanillaSolver(gg.WithLearnRate(0.1)),
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
	exp := thetas[0]
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

		// exp += mul
		exp, err = gg.Add(exp, mul)
		if err != nil {
			return nil, errors.Wrap(err, "x*theta addition error")
		}
	}

	// e^(theta0 + theta1*X1 + theta2*X2)
	eulerExp, err := gg.Exp(exp)
	if err != nil {
		return nil, errors.Wrap(err, "euler exponent error")
	}

	// 1+e^(theta0 + theta1*X1 + theta2*X2)
	divisor, err := gg.Add(gg.NewConstant(1.0), eulerExp)
	if err != nil {
		return nil, errors.Wrap(err, "euler exponent addition error")
	}

	// p = 1/(1+e^(theta0 + theta1*X1 + theta2*X2))
	p, err := gg.Div(gg.NewConstant(1.0), divisor)
	if err != nil {
		return nil, errors.Wrap(err, "prediction division error")
	}

	// compare predicted with actual
	se := gg.Must(gg.Square(gg.Must(gg.Sub(p, Y))))
	cost := gg.Must(gg.Mean(se))

	// back propagate data
	if _, err := gg.Grad(cost, thetas...); err != nil {
		return nil, errors.Wrap(err, "failed to propagate")
	}

	// new machine
	var machine gg.VM
	if m.useLisp {
		machine = gg.NewLispMachine(graph, gg.BindDualValues(thetas...))
	} else {
		machine = gg.NewTapeMachine(graph, gg.BindDualValues(thetas...))
	}
	defer machine.Close()

	// solve
	return solve(machine, thetas, m.iter, m.solver)
}

func solve(machine gg.VM, thetas []*gg.Node, iter int, solver gg.Solver) ([]float64, error) {
	model := make([]gg.ValueGrad, len(thetas))
	for i := range model {
		model[i] = thetas[i]
	}

	if gg.CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	var err error
	for i := 0; i < iter; i++ {
		if err = machine.RunAll(); err != nil {
			fmt.Printf("Error during iteration: %v: %v\n", i, err)
			break
		}

		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}

		machine.Reset() // Reset is necessary in a loop like this
	}

	thetaValues := make([]float64, len(thetas))
	for i := range thetas {
		thetaValues[i] = thetas[i].Value().Data().(float64)
	}
	return thetaValues, nil
}
