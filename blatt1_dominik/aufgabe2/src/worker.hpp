#pragma once

#include "barrier.hpp"
#include <vector>
#include <iostream>
#include <math.h>

class Worker {
public:


    //workers: Worker-Anzahl
    //id: ID des Workers
    //elements: Vektor mit Elementen
    //avg: Vektor mit Mittelwert pro Intervall (Size: _workers)
    //var: Vektor mit Varianz pro Intervall (Size: _workers)
    //global_avg: Globaler Mittelwert über alle Elemente in elements
    //global_var: Varianz aller Elemente in elements
    Worker(const int workers, const int id, std::vector<double>& e, std::vector<double>& avg, std::vector<double>& var, double& global_avg, double& global_var, Barrier& barrier) : id_(id), workers_(workers), elements_(e), avgResults_(avg), varResults_(var), avgResult_(global_avg), varResult_(global_var), barrier_(barrier) {
        bulkSize_ = e.size() / workers;

        start_ = id_ * bulkSize_;

        end_ = start_ + bulkSize_ + (id_ == (workers - 1)) * (e.size() - workers * bulkSize_);

    }

    void operator()() {
        avgResults_[id_] = local_average();
        barrier_.wait();
        double avg = global_average();
        varResults_[id_] = local_variance(avg);
        barrier_.wait();

        if (id_ == 0) {
            varResult_ = global_variance();
            avgResult_ = avg;
        }

        barrier_.wait();
    }

    double global_average() const {
        double avg = 0;

        for (double d : avgResults_) {
            avg += d;
        }

        return avg / avgResults_.size();
    }

    double local_average() const {
        double avg = 0;

        for (int i = start_; i < end_; i++) {
            avg += elements_[i];
        }

        return avg / (end_ - start_);
    }

    double local_variance(double avg) const {
        double var = 0;

        for (int i = start_; i < end_; i++) {
            var += (elements_[i] - avg) * (elements_[i] - avg);
        }

        return var / (elements_.size() - 1);
    }

    double global_variance() const {
        double var = 0;

        for (double d : varResults_) {
            var += d;
        }

        return var ;
    }

private:

    int id_;
    int workers_;
    int bulkSize_;
    int start_;
    int end_;
    std::vector<double>& elements_;
    std::vector<double>& avgResults_;
    std::vector<double>& varResults_;
    double& avgResult_;
    double& varResult_;

    Barrier& barrier_;
};


