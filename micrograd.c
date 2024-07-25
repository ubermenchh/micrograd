#include "micrograd.h" 

Value* init_value(float data) {
    Value* value = (Value*)malloc(sizeof(Value));
    if (value == NULL) 
        return NULL;

    value->data = data;
    value->grad = 0;
    value->_ctx = NULL;

    return value;
}

void free_value(Value* value) {
    if (value == NULL) return;
    if (value->_ctx != NULL) {
        if (value->_ctx->saved_values != NULL) {
            free(value->_ctx->saved_values);
        }
        free(value->_ctx);
    }
    free(value);
}

Context* init_context(void(*backward)(Context*, float), Value** saved_values, int value_count) {
    Context* ctx = (Context*)malloc(sizeof(Context));
    if (ctx == NULL) return NULL;

    ctx->backward = backward;
    ctx->saved_values = (Value**)malloc(value_count * sizeof(Value*));
    if (ctx->saved_values == NULL) {
        free(ctx); return NULL;
    }
    memcpy(ctx->saved_values, saved_values, value_count * sizeof(Value*));
    ctx->saved_value_count = value_count;
    return ctx;
}

void build_topo(Value* value, Value** visited, int* visited_count, Value** nodes, int* nodes_count) {
    bool found = false;
    for (int i = 0; i < *visited_count; i++) {
        if (visited[i] == value) {
            found = true;
            break;
        }
    }
    if (!found) {
        visited[*visited_count] = value;
        (*visited_count)++;
        if (value->_ctx) {
            for (int i = 0; i < value->_ctx->saved_value_count; i++) {
                build_topo(value->_ctx->saved_values[i], visited, visited_count, nodes, nodes_count);
            }
        }
        nodes[*nodes_count] = value;
        (*nodes_count)++;
    }
}

void backward(Value* value) {
    if (!value->_ctx) return;
    if (!value->grad) value->grad = 1.0;

    Value* visited[1000]; // assuming max of 1000 tensors 
    int visited_count = 0;
    Value* nodes[1000];
    int nodes_count = 0;
    build_topo(value, visited, &visited_count, nodes, &nodes_count);

    for (int i = nodes_count - 1; i >= 0; i--) {
        Value* t = nodes[i];
        if (t->_ctx) {
            t->_ctx->backward(t->_ctx, t->grad);
        }
    }
}

void print_value(Value* value) {
    printf("Value(data=%f, grad=%f)\n", value->data, value->grad);
}

Value* add_value(Value* a, Value* b) {
    Value* out = init_value(0.0);
    out->data = a->data + b->data;

    Value* saved_values[2] = {a, b};
    out->_ctx = init_context(add_backward, saved_values, 2);

    return out;
}

void add_backward(Context* ctx, float grad) {
    Value* a = ctx->saved_values[0];
    Value* b = ctx->saved_values[1];

    a->grad += grad;
    b->grad += grad;
}

Value* sub_value(Value* a, Value* b) {
    Value* out = init_value(0.0);
    out->data = a->data - b->data;

    Value* saved_values[2] = {a, b};
    out->_ctx = init_context(sub_backward, saved_values, 2);
    
    return out;
}

void sub_backward(Context* ctx, float grad) {
    Value* a = ctx->saved_values[0];
    Value* b = ctx->saved_values[1];

    a->grad += grad;
    b->grad += -grad;
}

Value* mul_value(Value* a, Value* b) {
    Value* out = init_value(0.0);
    out->data = a->data * b->data;

    Value* saved_values[2] = {a, b};
    out->_ctx = init_context(mul_backward, saved_values, 2);
    
    return out;
}

void mul_backward(Context* ctx, float grad) {
    Value* a = ctx->saved_values[0];
    Value* b = ctx->saved_values[1];

    a->grad += b->data * grad;
    b->grad += a->data * grad;
}

Value* div_value(Value* a, Value* b) {
    Value* out = init_value(0.0);
    out->data = a->data / b->data;

    Value* saved_values[2] = {a, b};
    out->_ctx = init_context(div_backward, saved_values, 2);

    return out;
}

void div_backward(Context* ctx, float grad) {
    Value* a = ctx->saved_values[0];
    Value* b = ctx->saved_values[1];

    a->grad += grad / b->data;
    b->grad += (grad * a->data) / (b->data*b->data);
}

Value* pow_value(Value* in, Value* power) {
    Value* out = init_value(0.0);
    out->data = pow(in->data, power->data);

    Value* saved_values[2] = {in, power};
    out->_ctx = init_context(pow_backward, saved_values, 2);

    return out;
}

void pow_backward(Context* ctx, float grad) {
    Value* in = ctx->saved_values[0];
    Value* power = ctx->saved_values[1];

    in->grad += (power->data * pow(in->data, power->data - 1)) * grad;
}

Value* relu_value(Value* in) {
    Value* out = init_value(0.0);
    if (in->data < 0) 
        out->data = 0;
    else 
        out->data = in->data;

    Value* saved_values[2] = {in, out};
    out->_ctx = init_context(relu_backward, saved_values, 2);
    return out;
}

void relu_backward(Context* ctx, float grad) {
    Value* in = ctx->saved_values[0];
    Value* out = ctx->saved_values[1];

    in->grad += (out->data > 0) * grad;
}

Value* tanh_value(Value* in) {
    Value* out = init_value(0.0);
    out->data = tanh(in->data);

    Value* saved_values[2] = {in, out};
    out->_ctx = init_context(tanh_backward, saved_values, 2);
    return out;
}

void tanh_backward(Context* ctx, float grad) {
    Value* in = ctx->saved_values[0];
    Value* out = ctx->saved_values[1];

    in->grad += (1 - (out->data*out->data)) * grad;
}

Value* exp_value(Value* in) {
    Value* out = init_value(0.0);
    out->data = exp(in->data);

    Value* saved_values[1] = {in};
    out->_ctx = init_context(exp_backward, saved_values, 1);
    return out;
}

void exp_backward(Context* ctx, float grad) {
    Value* in = ctx->saved_values[0];

    in->grad += exp(in->data) * grad;
}

Value* log_value(Value* in) {
    Value* out = init_value(0.0);
    out->data = log(in->data);

    Value* saved_values[1] = {in};
    out->_ctx = init_context(log_backward, saved_values, 1);
    return out;
}

void log_backward(Context* ctx, float grad) {
    Value* in = ctx->saved_values[0];

    in->grad += (1 / in->data) * grad;
}

float random_uniform() {
    return 2.f * ((float)rand() / RAND_MAX) - 1.f;
}

Neuron* init_neuron(int nin, bool nonlin) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (neuron == NULL) return NULL;
    neuron->base.impl = neuron;
    neuron->base.parameters = neuron_parameters;
    neuron->base.free = free_neuron;

    neuron->nin = nin;
    neuron->nonlin = nonlin;
    neuron->forward = neuron_forward;
    neuron->print = print_neuron;

    neuron->w = (Value**)malloc(nin * sizeof(Value*));
    if (neuron->w == NULL) {
        free(neuron);
        return NULL;
    } 

    float scale = sqrtf((float)nin);
    for (int i=0; i<nin; i++) {
        neuron->w[i] = init_value(random_uniform() * powf(nin, -0.5f));
        if (neuron->w[i] == NULL) {
            for (int j = 0; j < i; j++) 
                free(neuron->w[j]);
            free(neuron->w);
            free(neuron);
            return NULL;
        }
    }

    neuron->b = init_value(0.0);
    if (neuron->b == NULL) {
        for (int i=0; i < nin; i++) 
            free(neuron->w[i]);
        free(neuron->w);
        free(neuron);
        return NULL;
    }

    return neuron;
}

void free_neuron(Module* module) {
    Neuron* neuron = (Neuron*)module->impl;
    for (int i = 0; i < neuron->nin; i++) {
        free_value(neuron->w[i]);
    }
    free(neuron->w);
    free_value(neuron->b);
    free(neuron);
}

Value* neuron_forward(Neuron* neuron, Value** x) {
    Value* out = init_value(0.0);
    for (int i = 0; i < neuron->nin; i++) {
        out->data += mul_value(neuron->w[i], x[i])->data;
        out = add_value(out, neuron->b);
    }
    if (neuron->nonlin)
        tanh_value(out);
    return out;
}

Value** neuron_parameters(Module* module) {
    Neuron* neuron = (Neuron*)module->impl;
    Value** params = malloc((neuron->nin + 1) * sizeof(Value*)); // +1 for bias
    
    for (int i = 0; i < neuron->nin; i++) {
        params[i] = neuron->w[i];
    }
    params[neuron->nin] = neuron->b;
    return params;
}

void print_neuron(Neuron* neuron) {
    if (neuron->nonlin) {
        printf("Tanh_Neuron(%d)\n", neuron->nin);
    } else {
        printf("Linear_Neuron(%d)\n", neuron->nin);
    }
}

Layer* init_layer(int nin, int nout, bool nonlin) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (layer == NULL) return NULL;
    layer->base.impl = layer;
    layer->base.parameters = layer_parameters;
    layer->base.free = free_layer;
    
    layer->nin = nin;
    layer->nout = nout;
    layer->forward = layer_forward;
    layer->print = print_layer;

    layer->neurons = (Neuron**)malloc(nout * sizeof(Neuron*));
    if (layer->neurons == NULL) {
        free(layer);
        return NULL;
    }

    for (int i = 0; i < nout; i++) {
        layer->neurons[i] = init_neuron(nin, nonlin);
        if (layer->neurons[i] == NULL) {
            for (int j = 0; j < layer->nout; j++)
                free_neuron((Module*)layer->neurons[j]);
            free(layer->neurons);
            free(layer);
            return NULL;
        }
    }
    return layer;
}

void free_layer(Module* module) {
    Layer* layer = (Layer*)module->impl;
    for (int i = 0; i < layer->nin; i++) {
        free_neuron((Module*)layer->neurons[i]);
    }
    free(layer->neurons);
    free(layer);
}

Value** layer_forward(Layer* layer, Value** input) {
    Value** out = (Value**)malloc(layer->nout * sizeof(Value*));
    if (out == NULL) return NULL;

    for (int i = 0; i < layer->nout; i++) {
        out[i] = layer->neurons[i]->forward(layer->neurons[i], input);
        if (out[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free_value(out[j]);
            }
            free(out);
            return NULL;
        }
    }
    return out;
}

Value** layer_parameters(Module* module) {
    Layer* layer = (Layer*)module->impl;
    int total_params = 0;

    for (int i = 0; i < layer->nout; i++) {
        total_params += layer->nin + 1;
    }

    Value** params = (Value**)malloc(total_params * sizeof(Value*));
    if (params == NULL) return NULL;

    int param_idx = 0;
    for (int i = 0; i < layer->nout; i++) {
        Value** neuron_params = layer->neurons[i]->base.parameters((Module*)layer->neurons[i]);
        for (int j = 0; j < layer->nin + 1; j++) {
            params[param_idx++] = neuron_params[j];
        }
        free(neuron_params);
    }
    return params;
}

void print_layer(Layer* layer) {
    printf("Layer of [\n");
    for (int i = 0; i < layer->nout; i++) {
        printf("\t\t");
        print_neuron(layer->neurons[i]);
    }
    printf("\t]\n");
}

MLP* init_mlp(int nin, int nouts, bool nonlin) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    if (mlp == NULL) return NULL;
    
    mlp->base.impl = mlp;
    mlp->base.parameters = mlp_parameters;
    mlp->base.free = free_mlp;

    mlp->forward = mlp_forward;
    mlp->print = print_mlp;
    mlp->nin = nin;
    mlp->nouts = nouts;
    
    mlp->layers = (Layer**)malloc(nouts * sizeof(Layer*));
    if (mlp->layers == NULL) {
        free(mlp);
        return NULL;
    }
    
    for (int i = 0; i < nouts; i++) {
        mlp->layers[i] = init_layer(nin, nouts, nonlin);
        if (mlp->layers[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free_layer((Module*)mlp->layers[j]);
            }
            free(mlp->layers);
            free(mlp);
            return NULL;
        }
    }
    return mlp;
}

void free_mlp(Module* module) {
    MLP* mlp = (MLP*)module->impl;
    for (int i = 0; i < mlp->nouts; i++) {
        free_layer((Module*)mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

Value** mlp_forward(MLP* mlp, Value** input) {
    Value** out = input;
    Value** prev_out = NULL;
    if (out == NULL) return NULL;

    for (int i = 0; i < mlp->nouts; i++) {
        prev_out = out;
        out = mlp->layers[i]->forward(mlp->layers[i], out);
        if (out == NULL) 
            return NULL;

        if (i > 0) {
            for (int j = 0; j < mlp->layers[i-1]->nout; j++) {
                free_value(prev_out[j]);
            }
            free(prev_out);
        }
    }
    return out;
}

Value** mlp_parameters(Module* module) {
    MLP* mlp = (MLP*)module->impl;
    int total_params = 0;

    for (int i = 0; i < mlp->nouts; i++) {
        Value** layer_params = mlp->layers[i]->base.parameters((Module*)mlp->layers[i]);
        int layer_param_count = mlp->layers[i]->nin * mlp->layers[i]->nout + mlp->layers[i]->nout;
        total_params += layer_param_count;
        free(layer_params);
    }

    Value** params = (Value**)malloc(total_params * sizeof(Value*));
    if (params == NULL) return NULL;

    int param_index = 0;
    for (int i = 0; i < mlp->nouts; i++) {
        Value** layer_params = mlp->layers[i]->base.parameters((Module*)mlp->layers[i]);
        int layer_param_count = mlp->layers[i]->nin * mlp->layers[i]->nout + mlp->layers[i]->nout;
        for (int j = 0; j < layer_param_count; j++) {
            params[param_index++] = layer_params[j];
        }
        free(layer_params);
    }
    return params;
}

void print_mlp(MLP* mlp) {
    printf("MLP of {\n");
    for (int i = 0; i < mlp->nouts; i++) {
        printf("\t");
        print_layer(mlp->layers[i]);
    }
    printf("}\n");
}

int main(void) {
    Value* a = init_value(10.0);
    Value* b = init_value(3.0);
    MLP* mlp = init_mlp(10, 10, false);
    print_mlp(mlp);

    /*Value* out = neuron_forward(n, &a);
    backward(out);
    print_value(out);
    print_value(a); print_value(b);
    */

    free_value(a); free_value(b); //free_value(out);
}
