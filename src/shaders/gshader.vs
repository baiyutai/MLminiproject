#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 ourColor;

uniform mat4 transform;
uniform int goalInd;

void main()
{
	gl_Position = transform * vec4(aPos, 1.0);
    if (goalInd == 0){
        ourColor = vec3(1.0, 1.0, 0.0);
    }
	else{
        ourColor = vec3(0.0, 1.0, 1.0);
    }
}