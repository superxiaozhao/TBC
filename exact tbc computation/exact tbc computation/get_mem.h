#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <assert.h>

#define pid_t int

int get_phy_mem(const pid_t p)
{
	FILE *fd;
	int vmrss;
    char file[64] = {0};  
	char line_buff[256] = {0};
	char rdun1[32];
	char rdun2[32];
	
    sprintf(file, "/proc/%d/statm", p);
    fd = fopen (file, "r");
	char* ret = fgets(line_buff, sizeof(line_buff), fd);	
	sscanf(line_buff, "%s %d %s", rdun1, &vmrss, rdun2);
    fclose(fd);
	
    return vmrss;
}

int get_rmem(pid_t p)
{
    return get_phy_mem(p);
}