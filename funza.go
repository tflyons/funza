// package funza is a handy wrapper for linear and logistic regression using gorgonia
package funza

import (
	"runtime"

	"github.com/pkg/errors"
	gg "gorgonia.org/gorgonia"
)

// Option is an override the default option for logistic or linear regression
type Option func(*modeler) error

// WithVanillaSolver sets the regression to use the gorgonia vanilla solver
func WithVanillaSolver(rate float64) Option {
	return func(m *modeler) error {
		m.solver = gg.NewVanillaSolver(gg.WithLearnRate(rate))
		return nil
	}
}

// WithAdamSolver sets the regression to use the gorgonia adam solver
func WithAdamSolver(rate float64) Option {
	return func(m *modeler) error {
		m.solver = gg.NewAdamSolver(gg.WithLearnRate(rate))
		return nil
	}
}

// WithAdaGradSolver sets the regression to use the gorgonia ada grad solver
func WithAdaGradSolver(rate float64) Option {
	return func(m *modeler) error {
		m.solver = gg.NewAdaGradSolver(gg.WithLearnRate(rate))
		return nil
	}
}

// WithLispMachine sets the vm to use the gorgonia lisp machine
func WithLispMachine() Option {
	return func(m *modeler) error {
		m.useLisp = true
		return nil
	}
}

// WithIterations sets the number of iterations
func WithIterations(iter int) Option {
	return func(m *modeler) error {
		m.iter = iter
		return nil
	}
}

type modeler struct {
	iter    int
	solver  gg.Solver
	useLisp bool
}

func solve(graph *gg.ExprGraph, m *modeler, thetas []*gg.Node, predicted, actual *gg.Node) ([]float64, error) {
	// compare predicted with actual
	diff, err := gg.Sub(predicted, actual)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get diff")
	}
	se, err := gg.Square(diff)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get se")
	}
	cost, err := gg.Mean(se)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get cost")
	}

	// back propagate data
	if _, err = gg.Grad(cost, thetas...); err != nil {
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

	model := make([]gg.ValueGrad, len(thetas))
	for i := range model {
		model[i] = thetas[i]
	}

	if gg.CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}

	for i := 0; i < m.iter; i++ {
		if err = machine.RunAll(); err != nil {
			return nil, errors.Wrap(err, "error during solve iteration")
		}

		if err = m.solver.Step(model); err != nil {
			return nil, errors.Wrap(err, "error during solve step iteration")
		}

		machine.Reset() // Reset is necessary in a loop like this
	}

	thetaValues := make([]float64, len(thetas))
	for i := range thetas {
		thetaValues[i] = thetas[i].Value().Data().(float64)
	}
	return thetaValues, nil
}
