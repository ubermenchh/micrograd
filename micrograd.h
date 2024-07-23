#include <stdio.h> 
#include <math.h> 
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct Value Value;
typedef struct Context Context;

struct Value {
    float data;
    float grad;
    Context* _ctx;
};

struct Context {
    void (*backward) (Context* ctx, float grad);
    Value** saved_values;
    int saved_value_count;
};

Value* init_value(float data);
void free_value(Value* value);
Context* init_context(void(*backward)(Context*, float), Value** saved_values, int value_count);
void print_value(Value* value);

void backward(Value* value);

// Operations
Value* add_value(Value* a, Value* b);
void add_backward(Context* ctx, float grad);

Value* sub_value(Value* a, Value* b);
void sub_backward(Context* ctx, float grad);

Value* mul_value(Value* a, Value* b);
void mul_backward(Context* ctx, float grad);

Value* div_value(Value* a, Value* b);
void div_backward(Context* ctx, float grad);

Value* pow_value(Value* in, Value* power);
void pow_backward(Context* ctx, float grad);

Value* relu_value(Value* in);
void relu_backward(Context* ctx, float grad);

Value* tanh_value(Value* in);
void tanh_backward(Context* ctx, float grad);

Value* exp_value(Value* in);
void exp_backward(Context* ctx, float grad);

Value* log_value(Value* in);
void log_backward(Context* ctx, float grad);

// Layers 
typedef struct Module Module;
typedef struct Neuron Neuron;
typedef struct Layer Layer;
typedef struct MLP MLP;

struct Module {
    void* impl;
    Value** (*parameters)(Module* self);
    void (*free)(Module* self);
};

struct Neuron {
    Module base;
    int nin;
    bool nonlin;

    Value** w;
    Value* b;
    Value* (*forward)(Neuron* self, Value** input);
};

struct Layer {
    Module base;
    int nin;  // input dim
    int nout; // output dim

    Neuron** neurons;
    Value** (*forward)(Layer* self, Value** input);
}; 

struct MLP {
    Module base;
    int nin;
    int nouts;

    Layer** layers; 
    Value** (*forward)(MLP* self, Value** input);
};

Neuron* init_neuron(int nin, bool nonlin);
Value* neuron_forward(Neuron* Neuron, Value** x);
Value** neuron_parameters(Module* module);
void free_neuron(Module* module);

Layer* init_layer(int nin, int nouts, bool nonlin);
Value** layer_forward(Layer* layer, Value** x);
Value** layer_parameters(Module* module);
void free_layer(Module* module);

MLP* init_mlp(int nin, int nouts, bool nonlin);
Value** mlp_forward(MLP* mlp, Value** x);
Value** mlp_parameters(Module* module);
void free_mlp(Module* module);

