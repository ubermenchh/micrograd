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

void build_topo(Value* value, Value** visited, int visited_count, Value** nodes, int nodes_count) {
    bool found = false;
    for (int i = 0; i < visited_count; i++) {
        if (visited[i] == value) {
            found = true;
            break;
        }
    }
    if (!found) {
        visited[visited_count] = value;
        (visited_count)++;
        if (value->_ctx) {
            for (int i = 0; i < value->_ctx->saved_value_count; i++) {
                build_topo(value->_ctx->saved_values[i], visited, visited_count, nodes, nodes_count);
            }
        }
        nodes[nodes_count] = value;
        (nodes_count)++;
    }
}

void backward(Value* value) {
    if (!value->_ctx) return;
    if (!value->grad) value->grad = 1.0;

    Value* visited[1000]; // assuming max of 1000 tensors 
    int visited_count = 0;
    Value* nodes[1000];
    int nodes_count = 0;
    build_topo(value, visited, visited_count, nodes, nodes_count);

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


int main(void) {
    Value* a = init_value(10.0);
    Value* b = init_value(3.0);

    Value* out = mul_value(a, b);
    backward(out);
    print_value(out);
    print_value(a); print_value(b);

    free_value(a); free_value(b); free_value(out);
}
