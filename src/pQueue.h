
/*
 * A estrutura precisa ter 1 campo next.
 *
 */
#define PQ_QUEUE(a,b) \
    1
#define PQ_FIFO(a,b) \
    0
#define PQ_ADD_ELEMENT(head___,b,cmp) 												\
do {																			\
    __typeof(b) El;																\
    if(head___ == NULL) { b->next = NULL; head___ = b; break; }						\
    if (!cmp(head___,b)) { b->next = head___; head___ = b; break; } 						\
    El = head___; 																	\
    while(El != NULL) { 														\
        if(El->next != NULL)  {													\
            if(!cmp(El->next,b)) break;					 						\
        }																		\
        else break;																\
        El = El->next; 															\
    }																			\
    b->next = El->next;															\
    El->next = b;																\
}while(0)

#define PQ_REMOVE_FIRST(head___,elR)												\
do {																			\
    elR = head___;																	\
    if(head___ != NULL) head___ = head___->next;				 							\
}while(0)

#define PQ_PRINT(head___,prt) 														\
    do{																			\
        __typeof(head___) temp___;													\
        temp___ = head___; 															\
        if(temp___ == NULL) { printf("Empty!"); break; }							\
        while(temp___ != NULL) { prt(temp___); temp___ = temp___->next; } 					\
        printf("\n");                                                           \
    }while(0)

#define PQ_CONTAINS(head___,b,r,_cmp_)                                               \
    do {                                                                        \
    int pos___ = 0;                                                             \
    __typeof(head___) temp___;                                                        \
    temp___ = head___;                                                                \
    r = 0;                                                                      \
    while(temp___ != NULL) {                                                       \
        if(_cmp_(temp___,b)) { \
            r = pos___; \
            break;                                                                      \
        } \
        pos___++; \
        temp___ = temp___->next;                                                      \
    }                                                                           \
    }while(0)
