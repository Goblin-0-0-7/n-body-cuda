#include "particle.c"

// Setup
double N = 1000000; // Number of particles
double L = 10.0; // Length of the box (meter?)
double dt = 0.01; // Time step (second?)
double ep = 0.01; // softening factor
double G = 6.67430; // gravitational const. (10^-11 m³/kgs³)

int main(){
    double* particles = malloc(N * sizeof(particle));
}

void step(particle* part, particle* obj){
    double acc = 0;
    for(int i = 0; i < N; i++){
        double r_ij = 0; /* vector calc*/
        acc += part[i].mass * r_ij / pow((sqrt(r_ij) + sqrt(ep)), 3/2); /* stuff like 3/2 should be precalculated by compiler */
    }
    acc *= G;
}