#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader.h>

#include <iostream>
#include <math.h>
#include <Eigen/Eigen>
#include <vector>
#include <numeric>
#include <omp.h>
#include <time.h>
#include <random>

using namespace Eigen;
using namespace std;

#define PI 3.1415926535

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;
const unsigned int SCENE_SHRINK = 60;

float clamp(float x, float xmin, float xmax){
    if (x < xmin) return xmin;
    if (x > xmax) return xmax;
    return x;
}

float getRand(float rmin, float rmax){
    float num = (float)rand()/RAND_MAX;
    return num*(rmax-rmin)+rmin;
}

void drawCircle(float posx, float posy, float radius, int dim, Shader goalShader){
    goalShader.use();
    glm::mat4 transform;
    glm::vec3 trans_vec(0.0f, -1.0f, 0.0f);
    glm::vec3 shrink_vec(tanf(PI/dim)/tanf(PI/6)*radius/sqrt(3), radius/sqrt(3), 1.0f);
    for (int i = 0; i < dim; ++i){
        transform = glm::mat4(1.0f);
        transform = glm::translate(transform, glm::vec3(posx, posy, 0.0f));
        transform = glm::rotate(transform, (float)(2.0*PI*i/dim), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::scale(transform, shrink_vec);
        transform = glm::translate(transform, trans_vec);
        goalShader.setMat4("transform", transform);
        goalShader.setInt("goalInd", 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
}

bool collideTest(float posx, float posy, vector<vector<float>>& obstacles){
    if (abs(posx)>SCENE_SHRINK || abs(posy)>SCENE_SHRINK) return true;
    for (auto & obstacle : obstacles){
        float dx = obstacle[0]-posx, dy = obstacle[1] - posy;
        float dist = dx * dx + dy * dy;
        if (sqrt(dist) < obstacle[2]+1){
            return true;
        }
    }
    return false;
}

float collidePredict(float posx, float posy, float theta, vector<vector<float>>& obstacles){
    float obj_dist = 10000.0;
    if (cos(theta)==0.0){
        obj_dist = (sin(theta)>0.0) ? SCENE_SHRINK-posy:posy-SCENE_SHRINK;
    }
    else{
        obj_dist = (cos(theta)>0.0) ? (SCENE_SHRINK-posx)/abs(tan(theta)):(posx-SCENE_SHRINK)/abs(tan(theta));
    }
    obj_dist = clamp(obj_dist, 0.1, obj_dist);
    for (auto & obstacle : obstacles){
        float dx = obstacle[0]-posx, dy = obstacle[1] - posy;
        float dist = sqrt(dx * dx + dy * dy);
        float delta = atan2(dy, dx) - theta;
        if (cos(delta)<=0.0) continue;
        float vdist = abs(dist * sin(delta));
        if (vdist < obstacle[2]+1){
            obj_dist = clamp(dist*cos(delta)-sqrt((obstacle[2]+1)*(obstacle[2]+1)-vdist*vdist), 0.1, obj_dist);
        }
    }
    return obj_dist;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window, float& goalx, float& goaly, bool& goalNew)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS){
        goalNew = true;
        double tempx, tempy;
        glfwGetCursorPos(window, &tempx, &tempy);
        goalx = SCENE_SHRINK * (tempx - SCR_WIDTH/2) * 2/SCR_WIDTH;
        goaly = - (SCENE_SHRINK * (tempy - SCR_HEIGHT/2) * 2/SCR_HEIGHT);
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void two_layer_model(VectorXf& param, VectorXf& in_data, VectorXf& out_data, int hidden_size){
    // in -> hidden
    int in_size = in_data.size();
    Map<MatrixXf> w1(&param(0), hidden_size, in_size);
    Map<VectorXf> b1(&param(w1.size()), hidden_size);
    VectorXf hidden_layer = w1 * in_data + b1;
    # pragma omp parallel for
    for (int i = 0; i < hidden_layer.size(); ++i){
        hidden_layer(i) = (hidden_layer(i) >= 0) ? hidden_layer(i) : 0.1*hidden_layer(i); // leaky relu
    }
    // hidden -> out
    int out_size = out_data.size();
    Map<MatrixXf> w2(&param(w1.size()+b1.size()), out_size, hidden_size);
    Map<VectorXf> b2(&param(w1.size()+b1.size()+w2.size()), out_size);
    out_data = w2 * hidden_layer + b2;
}


void run_model(VectorXf& param, vector<float>& init_state, vector<vector<float>>& actions, vector<vector<float>>& states, float dt, float vmax, float omax, int hidden_size, vector<vector<float>>& obstacles, int steps, bool clampFlag = false){
    int in_size = init_state.size();
    VectorXf in_data = VectorXf::Zero(in_size);
    #pragma omp parallel for
    for (int i = 0; i < in_size; ++i)
        in_data(i) = init_state[i];
    int out_size = actions[0].size();
    VectorXf out_data = VectorXf::Zero(out_size);
    states[0] = vector<float>(init_state.begin(), init_state.begin()+4);
    for (int i = 0; i < steps; ++i){
        two_layer_model(param, in_data, out_data, hidden_size);
        if (clampFlag){
            actions[i][0] = clamp(out_data(0), -vmax, vmax); // velocity
            actions[i][1] = clamp(out_data(1), -omax, omax); // omega
        }
        else{
            actions[i][0] = out_data(0);
            actions[i][1] = out_data(1);
        }
        
        // update state
        states[i+1][0] = states[i][0] + actions[i][0]*dt*cos(states[i][2]); // xnew = x + v*dt*cos(theta)
        states[i+1][1] = states[i][1] + actions[i][0]*dt*sin(states[i][2]); // ynew = y + v*dt*cos(theta)
        states[i+1][2] = states[i][2] + actions[i][1]*dt; // anglenew = angle + omega*dt
        if (collideTest(states[i+1][0], states[i+1][1], obstacles))
            states[i+1][3] = 2.0;
        else if (collidePredict(states[i+1][0], states[i+1][1], states[i+1][2], obstacles) < 20.0)
            states[i+1][3] = 1.0;
        else
            states[i+1][3] = 0.0;
        for (int j = 0; j < 4; ++j){
            in_data(j) = states[i+1][j];
        }
    }
}

float reward(VectorXf& param, vector<float>& init_state, int steps, float time_total, float vmax, float omax, int hidden_size, vector<vector<float>> & obstacles){
    vector<vector<float>> actions(steps, vector<float>(2, 0));
    vector<vector<float>> states(steps+1, vector<float>(4, 0));
    run_model(param, init_state, actions, states, time_total/steps, vmax, omax, hidden_size, obstacles, actions.size());
    float reward = 0.0, dist = 0.0;
    float gamma = 0.9, gammaC = 1.0;
    for (int i = 1; i <= steps; ++i){
        float dx = states[i][0] - init_state[4];
        float dy = states[i][1] - init_state[5];
        dist = sqrt(dx*dx+dy*dy);
        float collide_dist = states[i][3], omega = actions[i-1][1], vel = actions[i-1][0];
        reward += gammaC * (100/(dist+0.1) - omega*omega - vel*vel/collide_dist);
        gammaC *= gamma;
    }
    return reward;
}


void cem(VectorXf& mu, VectorXf& th, vector<float>& init_state, int steps, float time_total, float vmax, float omax, int hidden_size, vector<vector<float>>& obstacles){
    int batch_size = 50;
    int iterations = 100;
    float elite_frac = 0.5;
    float init_std = 1.0;
    float noise_factor = 1.0;

    int elite_n = elite_frac * batch_size;

    int p_size = mu.size();
    VectorXf param = VectorXf::Zero(p_size);
    std::default_random_engine generator;
    for (int i = 0; i < iterations; ++i){
        MatrixXf params(p_size, batch_size);
        vector<float> score(batch_size);
        # pragma omp parallel for
        for (int j = 0; j < batch_size; ++j){
            # pragma omp parallel for
            for (int k = 0; k < p_size; ++k){
                normal_distribution<float> distribution(mu(k), th(k));
                param(k) = clamp(distribution(generator), mu(k)-3*th(k), mu(k)+3*th(k));
            }
            params.block(0, j, p_size, 1) = param;
            score[j] = reward(param, init_state, steps, time_total, vmax, omax, hidden_size, obstacles);
        }
        // sort
        vector<int> indices(batch_size);
        iota(indices.begin(), indices.end(), 0);
        stable_sort(indices.begin(), indices.end(), [&score](int i1, int i2){return score[i1]>score[i2];});
        // recompute mu and theta
        MatrixXf paramsCur(p_size, elite_n);
        # pragma omp parallel for
        for (int j = 0; j < elite_n; ++j){
            paramsCur.block(0, j, p_size, 1) = params.block(0, indices[j], p_size, 1);
        }
        mu = paramsCur.rowwise().mean();
        # pragma omp parallel for
        for (int j = 0; j < p_size; ++j){
            th(j) = (paramsCur.block(j, 0, 1, elite_n) - mu(j) * RowVectorXf::Ones(elite_n)).array().square().sum()/(elite_n - 1);
            th(j) = sqrt(th(j));
        }
        th = th + VectorXf::Ones(p_size)*(noise_factor/(i+1));
        float reward = 0.0;
        # pragma omp parallel for
        for (int j = 0; j < elite_n; ++j){
            reward += score[indices[j]]/elite_n;
        }
        if (i%50 == 0){
            cout<<"iter:"<<i<<"; reward_mean:"<<reward<<"; mu_mean:"<<mu.mean()<<"; th_mean:"<<th.mean()<<endl;
        }
    }
}

int main()
{
    srand((unsigned)time(NULL));
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "simple car model", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("D:\\InUseRepos\\MLMiniProject\\src\\shaders\\shader.vs", "D:\\InUseRepos\\MLMiniProject\\src\\shaders\\shader.fs");
    Shader goalShader("D:\\InUseRepos\\MLMiniProject\\src\\shaders\\gshader.vs", "D:\\InUseRepos\\MLMiniProject\\src\\shaders\\shader.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions           // texture coords
         0.0f,  1.0f, 0.0f,    1.0f, 1.0f, 0.0f,
         -0.866f,  -0.5f, 0.0f,    0.0f, 1.0f, 1.0f,
         0.866f, -0.5f, 0.0f,    1.0f, 0.0f, 1.0f,
    };
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // generate random obstacles
    int obs_num = 5;
    vector<vector<float>> obstacles(obs_num, vector<float>(3));
    #pragma omp parallel for
    for (auto & obstacle: obstacles){
        obstacle[0] = getRand(-(float)SCENE_SHRINK, (float)SCENE_SHRINK);
        obstacle[1] = getRand(-(float)SCENE_SHRINK, (float)SCENE_SHRINK);
        obstacle[2] = getRand(5, 10);
    }

    vector<float> init_state = {-50, 0, 0.01, 0.0, 50.0, 0.0};
    while (collideTest(init_state[0], init_state[1], obstacles)){
        init_state[0] = getRand(-(float)SCENE_SHRINK, (float)SCENE_SHRINK);
        init_state[1] = getRand(-(float)SCENE_SHRINK, (float)SCENE_SHRINK);
    }
    if (collideTest(init_state[0], init_state[1], obstacles))
        init_state[3] = 2.0;
    else if (collidePredict(init_state[0], init_state[1], init_state[2], obstacles) < 20.0)
        init_state[3] = 1.0;
    else
        init_state[3] = 0.0;
    int in_size = 6, hidden_size = 7, out_size = 2;
    float time_total = 0.1;
    int steps = 10.0;
    float vmax = 80.0, omax = 2.0;
    VectorXf mu = VectorXf::Zero((in_size+1)*hidden_size+(hidden_size+1)*out_size);
    VectorXf th = VectorXf::Ones((in_size+1)*hidden_size+(hidden_size+1)*out_size);
    vector<vector<float>> actions(steps, vector<float>(out_size, 0));
    vector<vector<float>> states(steps+1, vector<float>(4, 0));

    float nextGoalX = 0.0, nextGoalY = 0.0;
    bool nextGoalFlag = false, reachGoalFlag = false;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        processInput(window, nextGoalX, nextGoalY, nextGoalFlag);
        // computation
        // -----------
        if (nextGoalFlag && reachGoalFlag){
            init_state[4] = nextGoalX;
            init_state[5] = nextGoalY;
            nextGoalFlag = false;
            reachGoalFlag = false;
            mu = VectorXf::Zero((in_size+1)*hidden_size+(hidden_size+1)*out_size);
            th = VectorXf::Ones((in_size+1)*hidden_size+(hidden_size+1)*out_size);
        }
        if (!reachGoalFlag){
            cem(mu, th, init_state, steps, time_total, vmax, omax, hidden_size, obstacles);
            run_model(mu, init_state, actions, states, time_total/steps, vmax, omax, hidden_size, obstacles, 1, true);
            for (int i = 0; i < 4; ++i){
                init_state[i] = states[1][i];
            }
            float dx = init_state[0] - init_state[4], dy = init_state[1] - init_state[5];
            if (sqrt(dx*dx+dy*dy)<2.0){
                reachGoalFlag = true;
            }
        }

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(VAO);
        // draw current goal
        goalShader.use();
        glm::mat4 transform = glm::mat4(1.0f);
        transform = glm::translate(transform, glm::vec3(init_state[4]/SCENE_SHRINK, init_state[5]/SCENE_SHRINK, 0.0f));
        transform = glm::scale(transform, glm::vec3(1.0/SCENE_SHRINK, 1.0/SCENE_SHRINK, 1.0));
        goalShader.setMat4("transform", transform);
        goalShader.setInt("goalInd", 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // draw obstacles
        for (auto & obstacle:obstacles){
            drawCircle(obstacle[0]/SCENE_SHRINK, obstacle[1]/SCENE_SHRINK, obstacle[2]/SCENE_SHRINK, 10, goalShader);
        }
        // draw agent
        transform = glm::mat4(1.0f);
        transform = glm::translate(transform, glm::vec3(init_state[0]/SCENE_SHRINK, init_state[1]/SCENE_SHRINK, 0.0f));
        transform = glm::rotate(transform, init_state[2], glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::scale(transform, glm::vec3(1.0/SCENE_SHRINK, 1.0/SCENE_SHRINK, 1.0));
        ourShader.use();
        ourShader.setMat4("transform", transform);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // draw next goal
        if (nextGoalFlag){
            transform = glm::mat4(1.0f);
            transform = glm::translate(transform, glm::vec3(nextGoalX/SCENE_SHRINK, nextGoalY/SCENE_SHRINK, 0.0f));
            transform = scale(transform, glm::vec3(1.0/SCENE_SHRINK, 1.0/SCENE_SHRINK, 1.0));
            goalShader.use();
            goalShader.setMat4("transform", transform);
            goalShader.setInt("goalInd", 2);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}